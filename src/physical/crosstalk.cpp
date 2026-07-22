#include "physical/crosstalk.h"
#include <cmath>

namespace onn {

ThermalCrosstalk::ThermalCrosstalk(double coupling_strength, double sigma_spatial)
    : kappa_(coupling_strength), sigma_spatial_(sigma_spatial) {}

std::vector<double> ThermalCrosstalk::compute_phase_errors(
    const std::vector<MZIPosition>& positions,
    const std::vector<double>& phase_deltas) const {

    int N = static_cast<int>(positions.size());
    std::vector<double> errors(N, 0.0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            double dx = positions[i].col - positions[j].col;
            double dy = positions[i].row - positions[j].row;
            double dist_sq = dx * dx + dy * dy;
            double kernel = std::exp(-dist_sq / (2.0 * sigma_spatial_ * sigma_spatial_));
            errors[i] += kappa_ * kernel * phase_deltas[j];
        }
    }

    return errors;
}

} // namespace onn
