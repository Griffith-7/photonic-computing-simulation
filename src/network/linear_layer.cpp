#include "network/linear_layer.h"
#include <cmath>

namespace onn {

LinearLayer::LinearLayer(int in_features, int out_features,
                          MeshType mesh, bool physical,
                          const PhysicalConfig& phys_config)
    : in_features_(in_features),
      out_features_(out_features),
      N_(std::max(in_features, out_features)),
      mesh_type_(mesh),
      physical_(physical),
      physical_layer_(phys_config),
      phys_config_(phys_config) {
    W_ = Eigen::MatrixXcd::Zero(out_features, in_features);
    drift_state_ = Eigen::MatrixXcd::Zero(out_features, in_features);
    if (physical_) {
        rng_.seed(phys_config.seed);
        apply_drift_state();
    }
}

Eigen::VectorXcd LinearLayer::forward(const Eigen::VectorXcd& input) {
    if (!physical_) {
        return W_ * input;
    }

    int n = out_features_;
    Eigen::MatrixXcd W_eff = W_;

    if (phys_config_.drift_enabled) {
        W_eff = W_ + drift_state_;
    }

    Eigen::VectorXcd output = W_eff * input;

    if (phys_config_.attenuation_enabled) {
        double total_loss_db = phys_config_.coupling_loss_db * 2.0
                             + phys_config_.attenuation_db_per_cm * 0.01 * N_;
        double base_att = std::pow(10.0, -total_loss_db / 20.0);
        std::normal_distribution<double> att_noise(0.0, 0.03);
        for (int i = 0; i < n; i++) {
            double att = base_att * (1.0 + att_noise(rng_));
            output(i) *= att;
        }
    }

    if (phys_config_.shot_noise_enabled) {
        double expected_photons = phys_config_.quantum_efficiency * 1e6;
        for (int i = 0; i < n; i++) {
            double power = std::norm(output(i));
            double mean = expected_photons * power;
            if (mean > 0.1) {
                std::poisson_distribution<int> photon_dist(mean);
                int photons = photon_dist(rng_);
                double new_power = static_cast<double>(photons) / expected_photons;
                double old_power = power;
                if (old_power > 1e-20) {
                    output(i) *= std::sqrt(new_power / old_power);
                }
            }
        }
    }

    if (phys_config_.crosstalk_enabled) {
        double kappa = phys_config_.crosstalk_kappa;
        double sigma = phys_config_.crosstalk_sigma;
        Eigen::VectorXcd perturbed = output;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    double dist = std::abs(i - j);
                    double kernel = std::exp(-dist * dist / (2.0 * sigma * sigma));
                    perturbed(i) += kappa * kernel * output(j);
                }
            }
        }
        output = perturbed;
    }

    return output;
}

Eigen::VectorXcd LinearLayer::forward_ideal(const Eigen::VectorXcd& input) {
    return W_ * input;
}

void LinearLayer::set_weight_matrix(const Eigen::MatrixXcd& W) {
    W_ = W;
    mapping_valid_ = false;
    if (physical_) {
        apply_drift_state();
    }
}

void LinearLayer::reset_drift() {
    drift_state_ = Eigen::MatrixXcd::Zero(out_features_, in_features_);
}

void LinearLayer::apply_drift_state() {
    if (!phys_config_.drift_enabled) {
        drift_state_ = Eigen::MatrixXcd::Zero(out_features_, in_features_);
        return;
    }
    double drift_std = phys_config_.drift_sigma;
    std::normal_distribution<double> noise(0.0, drift_std);
    drift_state_ = Eigen::MatrixXcd::Zero(out_features_, in_features_);
    for (int r = 0; r < out_features_; r++) {
        for (int c = 0; c < in_features_; c++) {
            drift_state_(r, c) = Complex{noise(rng_), 0.0};
        }
    }
    drift_state_ = drift_state_.cwiseProduct(W_);
}

const OpticalMapping& LinearLayer::optical_mapping() const {
    ensure_mapping();
    return mapping_;
}

void LinearLayer::ensure_mapping() const {
    if (!mapping_valid_) {
        mapping_ = SVDMapper::map_to_optical(W_, mesh_type_);
        mapping_valid_ = true;
    }
}

} // namespace onn
