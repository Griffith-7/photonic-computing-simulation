#pragma once
#include "core/types.h"
#include "physical/drift.h"
#include "physical/attenuation.h"
#include "physical/shot_noise.h"
#include "physical/crosstalk.h"
#include "photonic/reck.h"
#include <vector>

namespace onn {

struct PhysicalConfig {
    bool drift_enabled = true;
    double drift_sigma = 0.05;

    bool attenuation_enabled = true;
    double attenuation_db_per_cm = 0.3;
    double coupling_loss_db = 0.1;

    bool shot_noise_enabled = false;
    double quantum_efficiency = 0.9;

    bool crosstalk_enabled = false;
    double crosstalk_kappa = 0.01;
    double crosstalk_sigma = 1.0;

    double time_step = 1.0;
    unsigned seed = 42;
};

class PhysicalLayer {
public:
    PhysicalLayer(const PhysicalConfig& config = PhysicalConfig{});

    Eigen::VectorXcd apply(Eigen::VectorXcd modes,
                            const std::vector<MZIRecord>& mzis,
                            const std::vector<MZIPosition>& positions);

    void advance_time(double dt);

    void reset();

    const PhysicalConfig& config() const { return config_; }
    void set_config(const PhysicalConfig& c) { config_ = c; }

private:
    PhysicalConfig config_;
    std::vector<PhaseDrift> drifts_;
    Attenuation attenuation_;
    ShotNoise shot_noise_;
    ThermalCrosstalk crosstalk_;
    double current_time_ = 0.0;
};

} // namespace onn
