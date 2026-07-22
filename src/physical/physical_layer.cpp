#include "physical/physical_layer.h"

namespace onn {

PhysicalLayer::PhysicalLayer(const PhysicalConfig& config)
    : config_(config),
      attenuation_(config.attenuation_db_per_cm, config.coupling_loss_db),
      shot_noise_(config.quantum_efficiency),
      crosstalk_(config.crosstalk_kappa, config.crosstalk_sigma) {
    RNG::instance().seed(config.seed);
    drifts_.resize(128);
}

Eigen::VectorXcd PhysicalLayer::apply(Eigen::VectorXcd modes,
                                       const std::vector<MZIRecord>& mzis,
                                       const std::vector<MZIPosition>& positions) {
    int N = modes.size();

    std::vector<double> phase_deltas(mzis.size(), 0.0);

    if (config_.drift_enabled) {
        for (size_t i = 0; i < mzis.size(); i++) {
            if (i >= drifts_.size()) drifts_.resize(i + 1);
            double drift_err = drifts_[i].get_phase_error(config_.time_step);
            phase_deltas[i] = drift_err;
        }
    }

    if (config_.crosstalk_enabled && !positions.empty() && mzis.size() == positions.size()) {
        auto crosstalk_errors = crosstalk_.compute_phase_errors(positions, phase_deltas);
        for (size_t i = 0; i < mzis.size(); i++) {
            phase_deltas[i] += crosstalk_errors[i];
        }
    }

    for (size_t i = 0; i < mzis.size(); i++) {
        const auto& m = mzis[i];
        double effective_theta = m.theta;
        double effective_phi = m.phi + phase_deltas[i];

        MZI mzi(effective_theta, effective_phi);

        if (config_.attenuation_enabled) {
            mzi.apply_with_loss(modes, m.mode1, m.mode2, config_.coupling_loss_db * 2);
        } else {
            mzi.apply(modes, m.mode1, m.mode2);
        }
    }

    if (config_.shot_noise_enabled) {
        for (int i = 0; i < N; i++) {
            double power = std::abs(modes(i));
            int photons = shot_noise_.sample(power, config_.time_step);
            double noisy_power = std::sqrt(static_cast<double>(photons));
            if (power > 1e-15) {
                modes(i) = modes(i) * (noisy_power / power);
            }
        }
    }

    return modes;
}

void PhysicalLayer::advance_time(double dt) {
    current_time_ += dt;
}

void PhysicalLayer::reset() {
    for (auto& d : drifts_) d.reset();
    current_time_ = 0.0;
}

} // namespace onn
