#include "data/mnist_loader.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>

namespace onn {

static uint32_t read_uint32_be(std::ifstream& f) {
    uint32_t val;
    f.read(reinterpret_cast<char*>(&val), 4);
    return ((val & 0xFF) << 24) | ((val & 0xFF00) << 8) |
           ((val >> 8) & 0xFF00) | ((val >> 24) & 0xFF);
}

static Eigen::MatrixXd read_images(const std::string& path, int& num_samples, int& image_size) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Failed to open: " << path << "\n";
        num_samples = 0;
        image_size = 784;
        return Eigen::MatrixXd();
    }

    uint32_t magic = read_uint32_be(f);
    num_samples = static_cast<int>(read_uint32_be(f));
    int rows = static_cast<int>(read_uint32_be(f));
    int cols = static_cast<int>(read_uint32_be(f));
    image_size = rows * cols;

    Eigen::MatrixXd images(num_samples, image_size);
    std::vector<unsigned char> buffer(num_samples * image_size);
    f.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < image_size; j++) {
            images(i, j) = static_cast<double>(buffer[i * image_size + j]) / 255.0;
        }
    }
    return images;
}

static Eigen::VectorXi read_labels(const std::string& path, int expected_count) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Failed to open: " << path << "\n";
        return Eigen::VectorXi::Zero(expected_count);
    }

    read_uint32_be(f);
    int count = static_cast<int>(read_uint32_be(f));

    Eigen::VectorXi labels(count);
    std::vector<unsigned char> buffer(count);
    f.read(reinterpret_cast<char*>(buffer.data()), count);

    for (int i = 0; i < count; i++) {
        labels(i) = static_cast<int>(buffer[i]);
    }
    return labels;
}

MNISTData MNISTLoader::load(const std::string& data_dir) {
    int num_train, image_size;
    Eigen::MatrixXd train_images = read_images(data_dir + "/train-images-idx3-ubyte",
                                                 num_train, image_size);
    Eigen::VectorXi train_labels = read_labels(data_dir + "/train-labels-idx1-ubyte",
                                                 num_train);

    return {train_images, train_labels, num_train, image_size};
}

MNISTData MNISTLoader::load_subset(const std::string& data_dir, int max_samples) {
    MNISTData full = load(data_dir);
    if (full.num_samples <= max_samples) return full;

    MNISTData subset;
    subset.num_samples = max_samples;
    subset.image_size = full.image_size;
    subset.images = full.images.topRows(max_samples);
    subset.labels = full.labels.head(max_samples);
    return subset;
}

Eigen::VectorXd MNISTLoader::normalize(const Eigen::VectorXd& image) {
    return image / 255.0;
}

Eigen::MatrixXd MNISTLoader::normalize_batch(const Eigen::MatrixXd& images) {
    return images / 255.0;
}

MNISTData MNISTLoader::generate_synthetic(int num_samples, int image_size,
                                          int num_classes, unsigned seed) {
    std::mt19937 rng(seed);

    std::vector<Eigen::VectorXd> centroids(num_classes, Eigen::VectorXd::Zero(image_size));
    for (int c = 0; c < num_classes; c++) {
        std::normal_distribution<double> dist(0.0, 1.0);
        for (int j = 0; j < image_size; j++) {
            centroids[c](j) = dist(rng);
        }
        centroids[c].normalize();
    }

    Eigen::MatrixXd images(num_samples, image_size);
    Eigen::VectorXi labels(num_samples);
    std::uniform_int_distribution<int> label_dist(0, num_classes - 1);
    std::normal_distribution<double> noise_dist(0.0, 0.15);

    for (int i = 0; i < num_samples; i++) {
        int label = label_dist(rng);
        labels(i) = label;
        for (int j = 0; j < image_size; j++) {
            images(i, j) = std::clamp(centroids[label](j) + noise_dist(rng), 0.0, 1.0);
        }
    }

    return {images, labels, num_samples, image_size};
}

} // namespace onn
