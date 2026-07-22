#pragma once
#include "core/types.h"
#include <Eigen/Dense>
#include <vector>

namespace onn {

struct MZIPosition {
    int row;
    int col;
};

class ThermalCrosstalk {
public:
    ThermalCrosstalk(double coupling_strength = 0.01, double sigma_spatial = 1.0);

    std::vector<double> compute_phase_errors(
        const std::vector<MZIPosition>& positions,
        const std::vector<double>& phase_deltas) const;

    double coupling_strength() const { return kappa_; }
    void set_coupling_strength(double k) { kappa_ = k; }

private:
    double kappa_;
    double sigma_spatial_;
};

} // namespace onn
