#pragma once
#include "core/types.h"
#include "photonic/reck.h"
#include <vector>
#include <Eigen/Dense>

namespace onn {

class ClementsDecomposer {
public:
    static std::vector<MZIRecord> decompose(const Eigen::MatrixXcd& U);

    static Eigen::MatrixXcd reconstruct(const std::vector<MZIRecord>& mzis, int N);

    static double error(const Eigen::MatrixXcd& U_target,
                        const std::vector<MZIRecord>& mzis);
};

} // namespace onn
