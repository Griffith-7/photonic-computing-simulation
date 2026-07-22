#pragma once
#include "core/types.h"
#include "photonic/svd_mapper.h"
#include "physical/physical_layer.h"
#include <Eigen/Dense>
#include <random>

namespace onn {

class LinearLayer {
public:
    LinearLayer(int in_features, int out_features,
                MeshType mesh = MeshType::CLEMENTS,
                bool physical = false,
                const PhysicalConfig& phys_config = PhysicalConfig{});

    Eigen::VectorXcd forward(const Eigen::VectorXcd& input);
    Eigen::VectorXcd forward_ideal(const Eigen::VectorXcd& input);

    const Eigen::MatrixXcd& weight_matrix() const { return W_; }
    void set_weight_matrix(const Eigen::MatrixXcd& W);
    void set_weight_only(const Eigen::MatrixXcd& W) { W_ = W; }

    const OpticalMapping& optical_mapping() const;
    void ensure_mapping() const;
    void set_optical_mapping(const OpticalMapping& m) { mapping_ = m; mapping_valid_ = true; }

    int in_features() const { return in_features_; }
    int out_features() const { return out_features_; }

    void reset_drift();
    void apply_drift_state();

private:
    int in_features_;
    int out_features_;
    int N_;
    MeshType mesh_type_;
    bool physical_;
    Eigen::MatrixXcd W_;
    mutable OpticalMapping mapping_;
    mutable bool mapping_valid_ = false;
    PhysicalLayer physical_layer_;
    PhysicalConfig phys_config_;
    std::mt19937 rng_;
    Eigen::MatrixXcd drift_state_;
};

} // namespace onn
