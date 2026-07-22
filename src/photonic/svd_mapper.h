#pragma once
#include "core/types.h"
#include "photonic/reck.h"
#include "photonic/clements.h"
#include <Eigen/Dense>
#include <vector>

namespace onn {

enum class MeshType { RECK, CLEMENTS };

struct SVDResult {
    Eigen::MatrixXcd U;
    Eigen::VectorXd S;
    Eigen::MatrixXcd Vh;
};

struct OpticalMapping {
    std::vector<MZIRecord> u_mzis;
    std::vector<double> singular_values;
    std::vector<MZIRecord> vh_mzis;
    Eigen::VectorXcd d_u;
    Eigen::VectorXcd d_v;
    Eigen::MatrixXcd U_rec;
    Eigen::MatrixXcd Vh_rec;
    int N;
    int N_out;
    MeshType mesh = MeshType::RECK;
};

class SVDMapper {
public:
    static SVDResult decompose(const Eigen::MatrixXcd& W);

    static OpticalMapping map_to_optical(const Eigen::MatrixXcd& W,
                                          MeshType mesh = MeshType::CLEMENTS);

    static Eigen::VectorXcd forward(const OpticalMapping& map,
                                     const Eigen::VectorXcd& input);

    static Eigen::VectorXcd forward_with_loss(const OpticalMapping& map,
                                               const Eigen::VectorXcd& input,
                                               double loss_per_mzi_db);
};

} // namespace onn
