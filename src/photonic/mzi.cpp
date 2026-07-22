#include "photonic/mzi.h"

namespace onn {

MZI::MZI(double theta, double phi) : theta_(theta), phi_(phi) {}

Matrix2cd MZI::transfer_matrix() const {
    Matrix2cd U;
    double ct = std::cos(theta_);
    double st = std::sin(theta_);
    U(0, 0) = ct;
    U(0, 1) = IM * st;
    U(1, 0) = IM * st;
    U(1, 1) = ct;

    Matrix2cd P;
    P(0, 0) = std::exp(IM * phi_ / 2.0);
    P(0, 1) = Complex{0};
    P(1, 0) = Complex{0};
    P(1, 1) = std::exp(-IM * phi_ / 2.0);

    return U * P;
}

void MZI::apply(VectorXcd& modes, int mode1, int mode2) const {
    Matrix2cd T = transfer_matrix();
    Complex e1 = modes(mode1);
    Complex e2 = modes(mode2);
    modes(mode1) = T(0, 0) * e1 + T(0, 1) * e2;
    modes(mode2) = T(1, 0) * e1 + T(1, 1) * e2;
}

void MZI::apply_with_loss(VectorXcd& modes, int mode1, int mode2,
                           double loss_per_mzi_db) const {
    apply(modes, mode1, mode2);
    double factor = std::pow(10.0, -loss_per_mzi_db / 20.0);
    modes(mode1) *= factor;
    modes(mode2) *= factor;
}

bool MZI::is_bar_state(double tol) const {
    return std::abs(std::fmod(theta_, PI)) < tol;
}

bool MZI::is_cross_state(double tol) const {
    return std::abs(std::fmod(theta_ + PI / 2.0, PI)) < tol;
}

MZI MZI::bar_state() { return MZI(0.0, 0.0); }
MZI MZI::cross_state() { return MZI(PI / 2.0, 0.0); }

} // namespace onn
