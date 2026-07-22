#include "physical/drift.h"

namespace onn {

PhaseDrift::PhaseDrift(double sigma) : sigma_(sigma) {}

double PhaseDrift::get_phase_error(double dt) {
    double noise = gaussian_noise(sigma_ * std::sqrt(dt));
    accumulated_ += noise;
    return accumulated_;
}

void PhaseDrift::reset() {
    accumulated_ = 0.0;
}

} // namespace onn
