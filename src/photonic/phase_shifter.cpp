#include "photonic/phase_shifter.h"
#include "core/types.h"

namespace onn {

PhaseShifter::PhaseShifter(double nominal_phase, double error_std)
    : nominal_phase_(nominal_phase),
      actual_phase_(nominal_phase),
      error_std_(error_std) {
    if (error_std_ > 0) {
        apply_error(error_std_);
    }
}

void PhaseShifter::set_nominal(double p) {
    nominal_phase_ = p;
    actual_phase_ = p;
    if (error_std_ > 0) {
        apply_error(error_std_);
    }
}

double PhaseShifter::voltage_to_phase(double voltage) const {
    return alpha_ * voltage * voltage + beta_ * voltage;
}

double PhaseShifter::phase_to_voltage(double phase) const {
    if (std::abs(alpha_) < 1e-15) {
        return (std::abs(beta_) > 1e-15) ? phase / beta_ : 0.0;
    }
    double disc = beta_ * beta_ + 4.0 * alpha_ * phase;
    if (disc < 0) return 0.0;
    return (-beta_ + std::sqrt(disc)) / (2.0 * alpha_);
}

void PhaseShifter::update(double dt, double drift_std) {
    double drift = gaussian_noise(drift_std * std::sqrt(dt));
    actual_phase_ += drift;
}

void PhaseShifter::apply_error(double error_std) {
    actual_phase_ = nominal_phase_ + gaussian_noise(error_std);
}

void PhaseShifter::reset() {
    actual_phase_ = nominal_phase_;
}

} // namespace onn
