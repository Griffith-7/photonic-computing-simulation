#pragma once
#include "core/types.h"
#include "network/linear_layer.h"
#include "network/activation.h"
#include "physical/drift_compensation.h"
#include <vector>
#include <Eigen/Dense>

namespace onn {

struct LayerConfig {
    int in_features;
    int out_features;
    ActivationType activation = ActivationType::RELU;
    MeshType mesh = MeshType::CLEMENTS;
};

class ONN {
public:
    ONN(const std::vector<LayerConfig>& layers, bool physical = false,
        const PhysicalConfig& phys_config = PhysicalConfig{});

    Eigen::VectorXd forward(const Eigen::VectorXd& input_real);
    Eigen::VectorXd forward_ideal(const Eigen::VectorXd& input_real);

    void load_weights(int layer_idx, const Eigen::MatrixXcd& W);

    void compensate(const Eigen::MatrixXd& X_ref, const Eigen::VectorXi& y_ref,
                    const CalibrationConfig& cal_config = CalibrationConfig{});

    int num_layers() const { return layers_.size(); }
    const LinearLayer& layer(int i) const { return layers_[i]; }
    LinearLayer& layer_mutable(int i) { return layers_[i]; }

private:
    std::vector<LinearLayer> layers_;
    std::vector<ActivationType> activations_;
    bool physical_;
};

} // namespace onn
