#pragma once
#include "core/types.h"
#include "photonic/mzi.h"
#include <vector>
#include <Eigen/Dense>

namespace onn {

struct MZIRecord {
    int mode1;
    int mode2;
    double theta;
    double phi;
};

class ReckDecomposer {
public:
    static std::vector<MZIRecord> decompose(const Eigen::MatrixXcd& U);

    static Eigen::MatrixXcd reconstruct(const std::vector<MZIRecord>& mzis, int N);

    static double error(const Eigen::MatrixXcd& U_target,
                        const std::vector<MZIRecord>& mzis);
};

} // namespace onn
