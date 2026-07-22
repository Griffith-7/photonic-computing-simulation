#pragma once
#include "core/types.h"
#include "photonic/reck.h"
#include <vector>
#include <Eigen/Dense>

namespace onn {

struct CalibrationConfig {
    double learning_rate = 0.1;
    double perturbation_delta = 0.01;
    int max_iterations = 50;
    double convergence_threshold = 1e-6;
};

class DriftCompensator {
public:
    DriftCompensator(const CalibrationConfig& config = CalibrationConfig{});

    std::vector<MZIRecord> calibrate(
        const std::vector<MZIRecord>& drifted_mzis,
        const Eigen::VectorXcd& input,
        const Eigen::VectorXcd& expected_output,
        int mesh_size);

    int iterations_used() const { return iterations_used_; }
    double final_error() const { return final_error_; }

private:
    CalibrationConfig config_;
    int iterations_used_ = 0;
    double final_error_ = 0.0;
};

} // namespace onn
