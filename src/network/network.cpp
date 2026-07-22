#include "network/network.h"
#include "photonic/reck.h"
#include <Eigen/Dense>

namespace onn {

ONN::ONN(const std::vector<LayerConfig>& configs, bool physical,
          const PhysicalConfig& phys_config)
    : physical_(physical) {
    for (const auto& cfg : configs) {
        layers_.emplace_back(cfg.in_features, cfg.out_features,
                             cfg.mesh, physical, phys_config);
        activations_.push_back(cfg.activation);
    }
}

Eigen::VectorXd ONN::forward(const Eigen::VectorXd& input_real) {
    Eigen::VectorXcd current = input_real.cast<Complex>();

    for (size_t i = 0; i < layers_.size(); i++) {
        current = layers_[i].forward(current);

        Eigen::VectorXd real_vals = current.array().real();
        Eigen::VectorXd activated = Activation::apply(real_vals, activations_[i]);
        current = activated.cast<Complex>();
    }

    if (activations_.back() != ActivationType::SOFTMAX) {
        Eigen::VectorXd real_vals = current.array().real();
        return Activation::softmax(real_vals);
    }

    return current.array().real();
}

Eigen::VectorXd ONN::forward_ideal(const Eigen::VectorXd& input_real) {
    Eigen::VectorXcd current = input_real.cast<Complex>();

    for (size_t i = 0; i < layers_.size(); i++) {
        current = layers_[i].forward_ideal(current);

        Eigen::VectorXd real_vals = current.array().real();
        Eigen::VectorXd activated = Activation::apply(real_vals, activations_[i]);
        current = activated.cast<Complex>();
    }

    if (activations_.back() != ActivationType::SOFTMAX) {
        Eigen::VectorXd real_vals = current.array().real();
        return Activation::softmax(real_vals);
    }

    return current.array().real();
}

void ONN::load_weights(int layer_idx, const Eigen::MatrixXcd& W) {
    layers_[layer_idx].set_weight_matrix(W);
}

void ONN::compensate(const Eigen::MatrixXd& X_ref, const Eigen::VectorXi& y_ref,
                      const CalibrationConfig& cal_config) {
    int num_samples = static_cast<int>(std::min<Eigen::Index>(X_ref.rows(), 5));
    int L = static_cast<int>(layers_.size());

    for (int li = 0; li < L; li++) {
        auto& lyr = layers_[li];
        const auto& W = lyr.weight_matrix();
        int fan_in = lyr.in_features();
        int fan_out = lyr.out_features();

        Eigen::MatrixXcd ideal_outs(fan_out, num_samples);
        Eigen::MatrixXcd phys_outs(fan_out, num_samples);
        Eigen::MatrixXcd inputs(fan_in, num_samples);

        for (int s = 0; s < num_samples; s++) {
            Eigen::VectorXcd current = X_ref.row(s).cast<Complex>();

            for (int k = 0; k < li; k++) {
                current = layers_[k].forward(current);
                Eigen::VectorXd rv = current.array().real();
                Eigen::VectorXd act = Activation::apply(rv, activations_[k]);
                current = act.cast<Complex>();
            }

            inputs.col(s) = current;
            ideal_outs.col(s) = W * current;
            phys_outs.col(s) = lyr.forward(current);
        }

        Eigen::MatrixXcd delta = phys_outs - ideal_outs;

        Eigen::MatrixXcd Xt = inputs.adjoint();
        Eigen::MatrixXcd XtX = Xt * inputs;
        Eigen::MatrixXcd pinv_X = XtX.inverse() * Xt;

        Eigen::MatrixXcd dW = delta * pinv_X;

        Eigen::MatrixXcd W_corrected = W;
        for (int r = 0; r < fan_out; r++) {
            for (int c = 0; c < fan_in; c++) {
                double w_val = std::abs(W(r, c));
                if (w_val > 1e-6) {
                    Complex scale = dW(r, c) / W(r, c);
                    double scale_real = scale.real();
                    if (std::abs(scale_real) < 0.5) {
                        W_corrected(r, c) = W(r, c) / (1.0 + scale_real);
                    }
                }
            }
        }

        lyr.set_weight_only(W_corrected);
        lyr.reset_drift();
    }
}

} // namespace onn
