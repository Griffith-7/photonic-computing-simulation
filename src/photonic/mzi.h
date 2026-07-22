#pragma once
#include "core/types.h"
#include <Eigen/Dense>

namespace onn {

using Matrix2cd = Eigen::Matrix2cd;
using VectorXcd = Eigen::VectorXcd;

class MZI {
public:
    MZI(double theta = 0.0, double phi = 0.0);

    double theta() const { return theta_; }
    double phi() const { return phi_; }
    void set_theta(double t) { theta_ = t; }
    void set_phi(double p) { phi_ = p; }

    Matrix2cd transfer_matrix() const;

    void apply(VectorXcd& modes, int mode1, int mode2) const;

    void apply_with_loss(VectorXcd& modes, int mode1, int mode2,
                         double loss_per_mzi_db) const;

    bool is_bar_state(double tol = 1e-10) const;
    bool is_cross_state(double tol = 1e-10) const;

    static MZI bar_state();
    static MZI cross_state();

private:
    double theta_;
    double phi_;
};

} // namespace onn
