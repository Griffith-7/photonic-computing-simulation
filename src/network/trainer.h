#pragma once
#include "core/types.h"
#include "network/network.h"
#include <vector>
#include <Eigen/Dense>

namespace onn {

struct TrainConfig {
    int epochs = 10;
    int batch_size = 32;
    double learning_rate = 0.001;
    std::vector<int> hidden_layers = {256};
    MeshType mesh = MeshType::CLEMENTS;

    int hidden_size() const {
        return hidden_layers.empty() ? 256 : hidden_layers[0];
    }
};

struct TrainResult {
    std::vector<double> train_loss;
    std::vector<double> test_accuracy;
    std::vector<Eigen::MatrixXd> all_weights;
    std::vector<ActivationType> activations;
    int input_size = 0;
    int num_classes = 10;

    const Eigen::MatrixXd& trained_weights_hidden() const { return all_weights[0]; }
    const Eigen::MatrixXd& trained_weights_output() const { return all_weights.back(); }
};

class Trainer {
public:
    static TrainResult train_digitall_twin(
        const Eigen::MatrixXd& X_train,
        const Eigen::VectorXi& y_train,
        const Eigen::MatrixXd& X_test,
        const Eigen::VectorXi& y_test,
        const TrainConfig& config);

    static ONN create_optical_network(
        const TrainResult& result,
        bool physical = false,
        const PhysicalConfig& phys_config = PhysicalConfig{});

    static ONN create_optical_network(
        const TrainResult& result,
        const std::vector<ActivationType>& activations,
        bool physical = false,
        const PhysicalConfig& phys_config = PhysicalConfig{});

    static int predict(ONN& network, const Eigen::VectorXd& input);
    static int predict_ideal(ONN& network, const Eigen::VectorXd& input);
    static double accuracy(ONN& network, const Eigen::MatrixXd& X, const Eigen::VectorXi& y);
    static double ideal_accuracy(ONN& network, const Eigen::MatrixXd& X, const Eigen::VectorXi& y);
};

} // namespace onn
