#include <gtest/gtest.h>
#include "core/types.h"
#include "physical/drift.h"
#include "physical/attenuation.h"
#include "physical/shot_noise.h"
#include "physical/crosstalk.h"
#include "physical/physical_layer.h"

using namespace onn;

TEST(DriftTest, ZeroMean) {
    PhaseDrift drift(0.02);
    double sum = 0.0;
    int N = 10000;
    for (int i = 0; i < N; i++) {
        sum += drift.get_phase_error(0.001);
    }
    double mean = sum / N;
    EXPECT_NEAR(mean, 0.0, 0.1);
}

TEST(DriftTest, Variance) {
    double dt = 0.001;
    double sigma = 0.02;
    double expected_var_per_step = sigma * sigma * dt;

    std::vector<double> samples;
    for (int trial = 0; trial < 200; trial++) {
        PhaseDrift d(sigma);
        for (int i = 0; i < 1000; i++) {
            d.get_phase_error(dt);
        }
        samples.push_back(d.get_phase_error(0));
    }

    double mean = 0.0;
    for (double s : samples) mean += s;
    mean /= samples.size();

    double var = 0.0;
    for (double s : samples) var += (s - mean) * (s - mean);
    var /= samples.size();

    double expected_var = 1000 * expected_var_per_step;
    double relative_error = std::abs(var - expected_var) / expected_var;
    EXPECT_LT(relative_error, 0.5) << "var=" << var << " expected=" << expected_var;
}

TEST(AttenuationTest, LossComputation) {
    Attenuation att(0.3, 0.1);
    double total = att.total_loss_db(4, 8, 1.0);
    EXPECT_GT(total, 0.0);
    EXPECT_NEAR(total, 0.3 * 1.0 + 4 * 0.1 + 8 * 0.05, 1e-10);
}

TEST(AttenuationTest, PowerReduction) {
    Attenuation att(3.0, 0.0);
    double power_in = 1.0;
    double power_out = att.apply(power_in);
    EXPECT_LT(power_out, power_in);
    EXPECT_GT(power_out, 0.0);
}

TEST(ShotNoiseTest, PoissonSampling) {
    ShotNoise sn(0.9);
    double power = 1e-6;
    double dt = 1e-12;

    long long total = 0;
    int N = 10000;
    for (int i = 0; i < N; i++) {
        total += sn.sample(power, dt);
    }
    double mean = static_cast<double>(total) / N;
    double expected = sn.expected_photons(power, dt);
    double relative_error = std::abs(mean - expected) / expected;
    EXPECT_LT(relative_error, 0.3) << "mean=" << mean << " expected=" << expected;
}

TEST(CrosstalkTest, AdjacentCoupling) {
    ThermalCrosstalk ct(0.1, 1.0);

    std::vector<MZIPosition> positions = {{0, 0}, {0, 1}, {0, 3}};
    std::vector<double> deltas = {0.1, 0.0, 0.0};

    auto errors = ct.compute_phase_errors(positions, deltas);

    EXPECT_NEAR(errors[0], 0.0, 1e-10);
    EXPECT_GT(std::abs(errors[1]), 0.0);
    EXPECT_GT(std::abs(errors[2]), 0.0);
    EXPECT_GT(std::abs(errors[1]), std::abs(errors[2]))
        << "Closer neighbor should have more crosstalk";
}

TEST(PhysicalLayerTest, EnableDisable) {
    PhysicalConfig cfg;
    cfg.drift_enabled = false;
    cfg.attenuation_enabled = false;
    cfg.shot_noise_enabled = false;
    cfg.crosstalk_enabled = false;

    PhysicalLayer layer(cfg);
    Eigen::VectorXcd modes(4);
    modes << Complex{1, 0}, Complex{0, 0}, Complex{0, 0}, Complex{0, 0};

    std::vector<MZIRecord> mzis = {{0, 1, PI / 4, 0.0}};
    std::vector<MZIPosition> positions = {{0, 0}};

    auto output = layer.apply(modes, mzis, positions);
    double total_power = output.squaredNorm();
    EXPECT_NEAR(total_power, 1.0, 1e-10);
}
