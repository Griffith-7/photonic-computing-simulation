#include "photonic/clements.h"

namespace onn {

std::vector<MZIRecord> ClementsDecomposer::decompose(const Eigen::MatrixXcd& U) {
    int N = U.rows();
    std::vector<MZIRecord> mzis;
    Eigen::MatrixXcd work = U;

    for (int col = 0; col < N - 1; col++) {
        for (int row = N - 1; row > col; row--) {
            int r1 = row - 1;
            int r2 = row;

            Complex a = work(r1, col);
            Complex b = work(r2, col);
            double mag_a = std::abs(a);
            double mag_b = std::abs(b);
            double r = std::sqrt(mag_a * mag_a + mag_b * mag_b);

            if (r < 1e-15) {
                mzis.push_back({r1, r2, 0.0, 0.0});
                continue;
            }

            double theta = std::atan2(mag_b, mag_a);
            double phi = 0.0;
            if (mag_a > 1e-15 && mag_b > 1e-15) {
                phi = std::arg(a) - std::arg(b) - PI / 2.0;
            } else if (mag_b > 1e-15) {
                phi = -PI / 2.0;
            }

            mzis.push_back({r1, r2, theta, phi});

            double ct = std::cos(theta);
            double st = std::sin(theta);

            for (int c = 0; c < N; c++) {
                Complex v0 = work(r1, c);
                Complex v1 = work(r2, c);
                work(r1, c) = ct * v0 + IM * std::exp(IM * phi) * st * v1;
                work(r2, c) = IM * std::exp(-IM * phi) * st * v0 + ct * v1;
            }
        }
    }

    return mzis;
}

Eigen::MatrixXcd ClementsDecomposer::reconstruct(const std::vector<MZIRecord>& mzis, int N) {
    return ReckDecomposer::reconstruct(mzis, N);
}

double ClementsDecomposer::error(const Eigen::MatrixXcd& U_target,
                                  const std::vector<MZIRecord>& mzis) {
    int N = U_target.rows();
    Eigen::MatrixXcd U_rec = reconstruct(mzis, N);
    Eigen::MatrixXcd product = U_rec.adjoint() * U_target;
    Eigen::MatrixXcd diag = product.diagonal().asDiagonal();
    return (product - diag).norm();
}

} // namespace onn
