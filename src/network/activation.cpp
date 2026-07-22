#include "network/activation.h"
#include <cmath>
#include <algorithm>

namespace onn {

Eigen::VectorXd Activation::softmax(const Eigen::VectorXd& x) {
    Eigen::VectorXd exp_x = (x.array() - x.maxCoeff()).exp();
    return exp_x / exp_x.sum();
}

Eigen::VectorXd Activation::relu(const Eigen::VectorXd& x) {
    return x.array().max(0.0);
}

Eigen::VectorXd Activation::sigmoid(const Eigen::VectorXd& x) {
    return 1.0 / (1.0 + (-x.array()).exp());
}

Eigen::VectorXd Activation::tanh_act(const Eigen::VectorXd& x) {
    return x.array().tanh();
}

Eigen::VectorXd Activation::apply(const Eigen::VectorXd& x, ActivationType type) {
    switch (type) {
        case ActivationType::RELU: return relu(x);
        case ActivationType::SIGMOID: return sigmoid(x);
        case ActivationType::TANH: return tanh_act(x);
        case ActivationType::SOFTMAX: return softmax(x);
        default: return x;
    }
}

Eigen::VectorXcd Activation::apply_optical(const Eigen::VectorXcd& x, ActivationType type) {
    Eigen::VectorXd real_vals = x.array().abs();
    Eigen::VectorXd activated = apply(real_vals, type);
    Eigen::VectorXcd result(x.size());
    for (int i = 0; i < x.size(); i++) {
        result(i) = Complex{activated(i), 0.0};
    }
    return result;
}

} // namespace onn
