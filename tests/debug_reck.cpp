#include <iostream>
#include <cmath>
#include "core/types.h"
#include "photonic/reck.h"
#include <Eigen/Dense>
#include <random>

using namespace onn;

int main() {
    int N = 3;
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    Eigen::MatrixXcd U(N, N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            U(i, j) = Complex{dist(rng), dist(rng)};
    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(U);
    U = qr.householderQ();

    std::cout << "Target U:\n" << U << "\n\n";

    auto mzis = ReckDecomposer::decompose(U);
    std::cout << "MZIs: " << mzis.size() << "\n";
    for (auto& m : mzis) {
        std::cout << "  (" << m.mode1 << "," << m.mode2 << ") theta=" << m.theta << " phi=" << m.phi << "\n";
    }

    auto U_rec = ReckDecomposer::reconstruct(mzis, N);
    std::cout << "\nReconstructed:\n" << U_rec << "\n\n";

    auto product = U_rec.adjoint() * U;
    std::cout << "U_rec† * U:\n" << product << "\n\n";

    auto diag = product.diagonal().asDiagonal();
    double err = (product - diag).norm();
    std::cout << "Error: " << err << "\n";

    std::cout << "\nDifference matrix:\n" << (U_rec - U) << "\n";

    return 0;
}
