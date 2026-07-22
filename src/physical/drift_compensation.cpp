#include "physical/drift_compensation.h"
#include "photonic/reck.h"
#include <cmath>
#include <random>

namespace onn {

DriftCompensator::DriftCompensator(const CalibrationConfig& config)
    : config_(config) {}

std::vector<MZIRecord> DriftCompensator::calibrate(
    const std::vector<MZIRecord>& drifted_mzis,
    const Eigen::VectorXcd& input,
    const Eigen::VectorXcd& expected_output,
    int mesh_size) {

    int K = static_cast<int>(drifted_mzis.size());
    std::vector<MZIRecord> current = drifted_mzis;
    std::mt19937 rng(42);
    std::bernoulli_distribution coin(0.5);

    auto compute_output = [&](const std::vector<MZIRecord>& mzis) {
        Eigen::MatrixXcd U_rec = ReckDecomposer::reconstruct(mzis, mesh_size);
        Eigen::VectorXcd modes = Eigen::VectorXcd::Zero(mesh_size);
        modes.head(input.size()) = input;
        return U_rec * modes;
    };

    auto output_error = [&](const Eigen::VectorXcd& out) {
        Eigen::VectorXcd diff = out.head(expected_output.size()) - expected_output;
        return diff.squaredNorm();
    };

    Eigen::VectorXcd expected = expected_output;
    double err0 = output_error(compute_output(current));

    for (int iter = 0; iter < config_.max_iterations; iter++) {
        double a = config_.learning_rate / (iter + 1.0);
        double c = config_.perturbation_delta / std::sqrt(iter + 1.0);

        std::vector<double> delta(K);
        for (int k = 0; k < K; k++) {
            delta[k] = coin(rng) ? 1.0 : -1.0;
        }

        std::vector<MZIRecord> mzi_plus = current;
        std::vector<MZIRecord> mzi_minus = current;
        for (int k = 0; k < K; k++) {
            mzi_plus[k].phi += c * delta[k];
            mzi_minus[k].phi -= c * delta[k];
        }

        double err_plus = output_error(compute_output(mzi_plus));
        double err_minus = output_error(compute_output(mzi_minus));

        for (int k = 0; k < K; k++) {
            double grad = (err_plus - err_minus) / (2.0 * c * delta[k]);
            current[k].phi -= a * grad;
        }

        double current_err = output_error(compute_output(current));
        iterations_used_ = iter + 1;
        final_error_ = current_err;

        if (current_err < config_.convergence_threshold) {
            break;
        }
    }

    return current;
}

} // namespace onn
