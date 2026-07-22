#pragma once
#include "core/types.h"
#include <Eigen/Dense>
#include <string>

namespace onn {

struct MNISTData {
    Eigen::MatrixXd images;
    Eigen::VectorXi labels;
    int num_samples;
    int image_size;
};

class MNISTLoader {
public:
    static MNISTData load(const std::string& data_dir);
    static MNISTData load_subset(const std::string& data_dir, int max_samples);

    static MNISTData generate_synthetic(int num_samples = 200, int image_size = 16,
                                        int num_classes = 10, unsigned seed = 42);

    static Eigen::VectorXd normalize(const Eigen::VectorXd& image);
    static Eigen::MatrixXd normalize_batch(const Eigen::MatrixXd& images);
};

} // namespace onn
