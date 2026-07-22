#include <gtest/gtest.h>
#include "core/types.h"
#include "photonic/mzi.h"

using namespace onn;

TEST(MZITest, Unitarity) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0, 2 * PI);

    for (int trial = 0; trial < 100; trial++) {
        MZI mzi(dist(rng), dist(rng));
        Matrix2cd U = mzi.transfer_matrix();
        Matrix2cd product = U.adjoint() * U;
        EXPECT_NEAR(product(0, 0).real(), 1.0, 1e-12);
        EXPECT_NEAR(product(1, 1).real(), 1.0, 1e-12);
        EXPECT_NEAR(std::abs(product(0, 1)), 0.0, 1e-12);
        EXPECT_NEAR(std::abs(product(1, 0)), 0.0, 1e-12);
    }
}

TEST(MZITest, BarState) {
    MZI bar = MZI::bar_state();
    VectorXcd in(2);
    in << Complex{1.0, 0.0}, Complex{0.0, 0.0};
    VectorXcd out = in;
    bar.apply(out, 0, 1);
    EXPECT_NEAR(std::abs(out(0)), 1.0, 1e-10);
    EXPECT_NEAR(std::abs(out(1)), 0.0, 1e-10);
}

TEST(MZITest, CrossState) {
    MZI cross = MZI::cross_state();
    VectorXcd in(2);
    in << Complex{1.0, 0.0}, Complex{0.0, 0.0};
    VectorXcd out = in;
    cross.apply(out, 0, 1);
    EXPECT_NEAR(std::abs(out(0)), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(out(1)), 1.0, 1e-10);
}

TEST(MZITest, PowerConservation) {
    MZI mzi(0.7, 1.3);
    VectorXcd in(2);
    in << Complex{3.0, 1.0}, Complex{1.0, -2.0};
    double input_power = in.squaredNorm();

    VectorXcd out = in;
    mzi.apply(out, 0, 1);
    double output_power = out.squaredNorm();

    EXPECT_NEAR(input_power, output_power, 1e-12);
}

TEST(MZITest, ArbitrarySplit) {
    for (double theta = 0; theta < PI; theta += 0.1) {
        MZI mzi(theta, 0.0);
        VectorXcd in(2);
        in << Complex{1.0, 0.0}, Complex{0.0, 0.0};
        VectorXcd out = in;
        mzi.apply(out, 0, 1);

        double power0 = std::norm(out(0));
        double power1 = std::norm(out(1));
        EXPECT_NEAR(power0 + power1, 1.0, 1e-12);
        EXPECT_NEAR(power0, std::cos(theta) * std::cos(theta), 1e-12);
        EXPECT_NEAR(power1, std::sin(theta) * std::sin(theta), 1e-12);
    }
}
