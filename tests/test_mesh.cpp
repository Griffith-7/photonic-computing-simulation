#include <gtest/gtest.h>
#include "core/types.h"
#include "photonic/reck.h"
#include "photonic/clements.h"
#include "photonic/svd_mapper.h"
#include <random>

using namespace onn;

static Eigen::MatrixXcd random_unitary(int N, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::MatrixXcd U(N, N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            U(i, j) = Complex{dist(rng), dist(rng)};
    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(U);
    return qr.householderQ();
}

TEST(ReckTest, Decompose4x4) {
    Eigen::MatrixXcd U = random_unitary(4, 42);
    auto mzis = ReckDecomposer::decompose(U);
    double err = ReckDecomposer::error(U, mzis);
    EXPECT_LT(err, 1e-8) << "Reck 4x4 decomposition error too large";
}

TEST(ReckTest, Decompose8x8) {
    Eigen::MatrixXcd U = random_unitary(8, 123);
    auto mzis = ReckDecomposer::decompose(U);
    double err = ReckDecomposer::error(U, mzis);
    EXPECT_LT(err, 1e-6) << "Reck 8x8 decomposition error too large";
}

TEST(ReckTest, Decompose16x16) {
    Eigen::MatrixXcd U = random_unitary(16, 456);
    auto mzis = ReckDecomposer::decompose(U);
    double err = ReckDecomposer::error(U, mzis);
    EXPECT_LT(err, 1e-4) << "Reck 16x16 decomposition error too large";
}

TEST(ClementsTest, Decompose4x4) {
    Eigen::MatrixXcd U = random_unitary(4, 42);
    auto mzis = ClementsDecomposer::decompose(U);
    double err = ClementsDecomposer::error(U, mzis);
    EXPECT_LT(err, 1e-8) << "Clements 4x4 decomposition error too large";
}

TEST(ClementsTest, Decompose8x8) {
    Eigen::MatrixXcd U = random_unitary(8, 123);
    auto mzis = ClementsDecomposer::decompose(U);
    double err = ClementsDecomposer::error(U, mzis);
    EXPECT_LT(err, 1e-6) << "Clements 8x8 decomposition error too large";
}

TEST(ClementsTest, Decompose16x16) {
    Eigen::MatrixXcd U = random_unitary(16, 456);
    auto mzis = ClementsDecomposer::decompose(U);
    double err = ClementsDecomposer::error(U, mzis);
    EXPECT_LT(err, 1e-4) << "Clements 16x16 decomposition error too large";
}

TEST(SVDTest, DecomposeArbitraryMatrix) {
    int N = 8;
    Eigen::MatrixXcd W = Eigen::MatrixXcd::Random(N, N);

    auto svd = SVDMapper::decompose(W);
    Eigen::MatrixXcd reconstructed = svd.U * svd.S.asDiagonal() * svd.Vh;
    double err = (W - reconstructed).norm() / W.norm();
    EXPECT_LT(err, 1e-12);
}

TEST(SVDTest, OpticalForwardPass) {
    int N = 8;
    Eigen::MatrixXcd W = Eigen::MatrixXcd::Random(N, N);
    auto mapping = SVDMapper::map_to_optical(W, MeshType::RECK);

    Eigen::VectorXcd input = Eigen::VectorXcd::Random(N);
    auto output = SVDMapper::forward(mapping, input);
    Eigen::VectorXcd expected = W * input;

    double err = (output - expected).norm() / expected.norm();
    EXPECT_LT(err, 1e-8);
}

TEST(SVDTest, NonSquareMatrix) {
    int in = 16, out = 8;
    Eigen::MatrixXcd W = Eigen::MatrixXcd::Random(out, in);
    auto mapping = SVDMapper::map_to_optical(W, MeshType::RECK);

    Eigen::VectorXcd input = Eigen::VectorXcd::Random(in);
    auto output = SVDMapper::forward(mapping, input);
    Eigen::VectorXcd expected = W * input;

    double err = (output.head(out) - expected).norm() / expected.norm();
    EXPECT_LT(err, 1e-8);
}

TEST(SVDTest, ClementsOpticalForwardPass) {
    int N = 8;
    Eigen::MatrixXcd W = Eigen::MatrixXcd::Random(N, N);
    auto mapping = SVDMapper::map_to_optical(W, MeshType::CLEMENTS);

    Eigen::VectorXcd input = Eigen::VectorXcd::Random(N);
    auto output = SVDMapper::forward(mapping, input);
    Eigen::VectorXcd expected = W * input;

    double err = (output - expected).norm() / expected.norm();
    EXPECT_LT(err, 1e-8);
}

TEST(SVDTest, ClementsNonSquare) {
    int in = 16, out = 8;
    Eigen::MatrixXcd W = Eigen::MatrixXcd::Random(out, in);
    auto mapping = SVDMapper::map_to_optical(W, MeshType::CLEMENTS);

    Eigen::VectorXcd input = Eigen::VectorXcd::Random(in);
    auto output = SVDMapper::forward(mapping, input);
    Eigen::VectorXcd expected = W * input;

    double err = (output.head(out) - expected).norm() / expected.norm();
    EXPECT_LT(err, 1e-8);
}
