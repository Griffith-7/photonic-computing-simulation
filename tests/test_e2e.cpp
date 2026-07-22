#include <gtest/gtest.h>
#include "core/types.h"
#include "network/trainer.h"
#include "network/network.h"
#include "data/mnist_loader.h"
#include <chrono>
#include <iostream>
#include <Eigen/Dense>

using namespace onn;

TEST(E2ETest, RealMNISTLoad) {
    MNISTData train = MNISTLoader::load("data/mnist");
    EXPECT_GT(train.num_samples, 50000);
    EXPECT_EQ(train.image_size, 784);

    MNISTData test = MNISTLoader::load("data/mnist");
    EXPECT_GT(test.num_samples, 9000);
}

TEST(E2ETest, RealMNISTDigitalTwin) {
    MNISTData train = MNISTLoader::load("data/mnist");
    MNISTData test = MNISTLoader::load_subset("data/mnist", 1000);

    TrainConfig cfg;
    cfg.epochs = 10;
    cfg.batch_size = 128;
    cfg.learning_rate = 0.01;
    cfg.hidden_layers = {256, 128};

    auto t0 = std::chrono::high_resolution_clock::now();
    TrainResult result = Trainer::train_digitall_twin(
        train.images, train.labels,
        test.images, test.labels, cfg);
    auto t1 = std::chrono::high_resolution_clock::now();
    double train_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double digital_acc = result.test_accuracy.back();
    std::cout << "  Training time: " << train_ms << " ms\n";
    std::cout << "  Digital accuracy: " << digital_acc * 100.0 << "%\n";

    EXPECT_GT(digital_acc, 0.90) << "Digital twin should reach >90% on MNIST";
}

TEST(E2ETest, RealMNISTOpticalInference) {
    MNISTData train = MNISTLoader::load("data/mnist");
    MNISTData test = MNISTLoader::load_subset("data/mnist", 500);

    TrainConfig cfg;
    cfg.epochs = 10;
    cfg.batch_size = 128;
    cfg.learning_rate = 0.01;
    cfg.hidden_layers = {256, 128};

    TrainResult result = Trainer::train_digitall_twin(
        train.images, train.labels,
        test.images, test.labels, cfg);

    double digital_acc = result.test_accuracy.back();

    ONN network = Trainer::create_optical_network(result, false);

    double ideal_acc = Trainer::ideal_accuracy(network, test.images, test.labels);

    auto t0 = std::chrono::high_resolution_clock::now();
    double optical_acc = Trainer::accuracy(network, test.images, test.labels);
    auto t1 = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  Digital accuracy: " << digital_acc * 100.0 << "%\n";
    std::cout << "  Ideal accuracy (forward_ideal): " << ideal_acc * 100.0 << "%\n";
    std::cout << "  Optical accuracy (mesh forward): " << optical_acc * 100.0 << "%\n";
    std::cout << "  Inference time: " << infer_ms << " ms (" << test.num_samples << " samples)\n";

    EXPECT_GT(ideal_acc, 0.10) << "Ideal network should produce valid predictions";
}

TEST(E2ETest, RealMNISTPhysicalWithCompensation) {
    MNISTData train = MNISTLoader::load("data/mnist");
    MNISTData test = MNISTLoader::load_subset("data/mnist", 200);

    TrainConfig cfg;
    cfg.epochs = 5;
    cfg.batch_size = 128;
    cfg.learning_rate = 0.01;
    cfg.hidden_layers = {128};

    TrainResult result = Trainer::train_digitall_twin(
        train.images, train.labels,
        test.images, test.labels, cfg);

    PhysicalConfig phys;
    phys.drift_enabled = true;
    phys.drift_sigma = 0.10;
    phys.attenuation_enabled = true;
    phys.shot_noise_enabled = false;
    phys.crosstalk_enabled = false;

    ONN ideal_net = Trainer::create_optical_network(result, false);
    ONN phys_net = Trainer::create_optical_network(result, true, phys);

    double ideal_acc = Trainer::ideal_accuracy(ideal_net, test.images, test.labels);
    double phys_acc_before = Trainer::accuracy(phys_net, test.images, test.labels);

    Eigen::MatrixXd X_ref = train.images.topRows(100);
    Eigen::VectorXi y_ref = train.labels.head(100);

    CalibrationConfig cal_cfg;
    cal_cfg.max_iterations = 30;
    cal_cfg.learning_rate = 0.1;
    cal_cfg.perturbation_delta = 0.01;

    phys_net.compensate(X_ref, y_ref, cal_cfg);

    double phys_acc_after = Trainer::accuracy(phys_net, test.images, test.labels);

    std::cout << "  Ideal accuracy: " << ideal_acc * 100.0 << "%\n";
    std::cout << "  Physical (before comp): " << phys_acc_before * 100.0 << "%\n";
    std::cout << "  Physical (after comp): " << phys_acc_after * 100.0 << "%\n";
}
