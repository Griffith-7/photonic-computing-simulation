#include "photonic/svd_mapper.h"
#include <Eigen/SVD>

namespace onn {

namespace {

Eigen::MatrixXcd reconstruct_mzis(MeshType mesh, const std::vector<MZIRecord>& mzis, int N) {
    if (mesh == MeshType::CLEMENTS) {
        return ClementsDecomposer::reconstruct(mzis, N);
    }
    return ReckDecomposer::reconstruct(mzis, N);
}

} // anonymous namespace

SVDResult SVDMapper::decompose(const Eigen::MatrixXcd& W) {
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    SVDResult result;
    result.U = svd.matrixU();
    result.S = svd.singularValues();
    result.Vh = svd.matrixV().adjoint();
    return result;
}

OpticalMapping SVDMapper::map_to_optical(const Eigen::MatrixXcd& W, MeshType mesh) {
    SVDResult svd = decompose(W);
    int N = std::max(W.rows(), W.cols());

    std::vector<MZIRecord> u_mzis, vh_mzis;

    if (mesh == MeshType::RECK) {
        u_mzis = ReckDecomposer::decompose(svd.U);
        vh_mzis = ReckDecomposer::decompose(svd.Vh);
    } else {
        u_mzis = ClementsDecomposer::decompose(svd.U);
        vh_mzis = ClementsDecomposer::decompose(svd.Vh);
    }

    Eigen::MatrixXcd U_rec = reconstruct_mzis(mesh, u_mzis, svd.U.rows());
    Eigen::MatrixXcd Vh_rec = reconstruct_mzis(mesh, vh_mzis, svd.Vh.rows());

    Eigen::VectorXcd D_u = (U_rec.adjoint() * svd.U).diagonal();
    Eigen::VectorXcd D_v = (Vh_rec.adjoint() * svd.Vh).diagonal();

    OpticalMapping map;
    map.u_mzis = std::move(u_mzis);
    map.singular_values.resize(svd.S.size());
    for (int i = 0; i < svd.S.size(); i++) map.singular_values[i] = svd.S(i);
    map.vh_mzis = std::move(vh_mzis);
    map.N = N;
    map.N_out = W.rows();
    map.d_u = D_u;
    map.d_v = D_v;
    map.U_rec = std::move(U_rec);
    map.Vh_rec = std::move(Vh_rec);
    map.mesh = mesh;

    return map;
}

Eigen::VectorXcd SVDMapper::forward(const OpticalMapping& map,
                                     const Eigen::VectorXcd& input) {
    int N = map.N;
    Eigen::VectorXcd modes = Eigen::VectorXcd::Zero(N);
    modes.head(input.size()) = input;

    int vh_size = map.Vh_rec.rows();
    for (int i = 0; i < input.size() && i < vh_size && i < N; i++) {
        modes(i) *= map.d_v(i);
    }

    if (vh_size == N) {
        modes = map.Vh_rec * modes;
    } else {
        Eigen::MatrixXcd Vh_padded = Eigen::MatrixXcd::Identity(N, N);
        Vh_padded.topLeftCorner(vh_size, map.Vh_rec.cols()) = map.Vh_rec;
        modes = Vh_padded * modes;
    }

    for (int i = 0; i < map.N_out && i < (int)map.singular_values.size(); i++) {
        modes(i) *= map.singular_values[i];
    }

    for (int i = 0; i < map.N_out && i < (int)map.d_u.size(); i++) {
        modes(i) *= map.d_u(i);
    }

    modes.head(map.N_out) = map.U_rec * modes.head(map.N_out);

    return modes;
}

Eigen::VectorXcd SVDMapper::forward_with_loss(const OpticalMapping& map,
                                                const Eigen::VectorXcd& input,
                                                double loss_per_mzi_db) {
    int N = map.N;
    Eigen::VectorXcd modes = Eigen::VectorXcd::Zero(N);
    modes.head(input.size()) = input;

    int vh_size = map.Vh_rec.rows();
    for (int i = 0; i < input.size() && i < vh_size && i < N; i++) {
        modes(i) *= map.d_v(i);
    }

    if (vh_size == N) {
        modes = map.Vh_rec * modes;
    } else {
        Eigen::MatrixXcd Vh_padded = Eigen::MatrixXcd::Identity(N, N);
        Vh_padded.topLeftCorner(vh_size, map.Vh_rec.cols()) = map.Vh_rec;
        modes = Vh_padded * modes;
    }

    double vh_loss = std::pow(10.0, -loss_per_mzi_db * map.vh_mzis.size() / 20.0);
    modes *= vh_loss;

    for (int i = 0; i < map.N_out && i < (int)map.singular_values.size(); i++) {
        modes(i) *= map.singular_values[i];
    }

    for (int i = 0; i < map.N_out && i < (int)map.d_u.size(); i++) {
        modes(i) *= map.d_u(i);
    }

    modes.head(map.N_out) = map.U_rec * modes.head(map.N_out);

    double u_loss = std::pow(10.0, -loss_per_mzi_db * map.u_mzis.size() / 20.0);
    modes.head(map.N_out) *= u_loss;

    return modes;
}

} // namespace onn
