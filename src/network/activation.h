#pragma once
#include "core/types.h"
#include <Eigen/Dense>

namespace onn {

enum class ActivationType { RELU, SIGMOID, TANH, SOFTMAX, NONE };

class Activation {
public:
    static Eigen::VectorXd apply(const Eigen::VectorXd& x, ActivationType type);
    static Eigen::VectorXcd apply_optical(const Eigen::VectorXcd& x, ActivationType type);
    static Eigen::VectorXd softmax(const Eigen::VectorXd& x);
    static Eigen::VectorXd relu(const Eigen::VectorXd& x);
    static Eigen::VectorXd sigmoid(const Eigen::VectorXd& x);
    static Eigen::VectorXd tanh_act(const Eigen::VectorXd& x);
};

} // namespace onn
