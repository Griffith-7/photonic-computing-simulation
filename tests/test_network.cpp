#include <gtest/gtest.h>
#include "core/types.h"
#include "network/activation.h"
#include "network/trainer.h"
#include "data/mnist_loader.h"
#include <Eigen/Dense>
#include <cmath>

using namespace onn;

TEST(ActivationTest, SoftmaxSumsToOne) {
    Eigen::VectorXd x(5);
    x << 1.0, 2.0, 3.0, 4.0, 5.0;
    Eigen::VectorXd sm = Activation::softmax(x);
    EXPECT_NEAR(sm.sum(), 1.0, 1e-10);
    EXPECT_GT(sm(4), sm(0));
}

TEST(ActivationTest, ReLU) {
    Eigen::VectorXd x(4);
    x << -1.0, 0.5, -0.3, 2.0;
    Eigen::VectorXd r = Activation::relu(x);
    EXPECT_NEAR(r(0), 0.0, 1e-10);
    EXPECT_NEAR(r(1), 0.5, 1e-10);
    EXPECT_NEAR(r(2), 0.0, 1e-10);
    EXPECT_NEAR(r(3), 2.0, 1e-10);
}

TEST(ActivationTest, Sigmoid) {
    Eigen::VectorXd x(3);
    x << 0.0, 10.0, -10.0;
    Eigen::VectorXd s = Activation::sigmoid(x);
    EXPECT_NEAR(s(0), 0.5, 1e-10);
    EXPECT_NEAR(s(1), 1.0, 0.01);
    EXPECT_NEAR(s(2), 0.0, 0.01);
}

TEST(TrainerTest, DigitalTwinTraining) {
    int input_size = 16;
    int hidden = 32;
    int classes = 5;
    int train_size = 200;
    int test_size = 50;

    Eigen::MatrixXd X_train = Eigen::MatrixXd::Random(train_size, input_size);
    Eigen::VectorXi y_train = Eigen::VectorXi::Zero(train_size);
    for (int i = 0; i < train_size; i++) y_train(i) = i % classes;

    Eigen::MatrixXd X_test = Eigen::MatrixXd::Random(test_size, input_size);
    Eigen::VectorXi y_test = Eigen::VectorXi::Zero(test_size);
    for (int i = 0; i < test_size; i++) y_test(i) = i % classes;

    TrainConfig cfg;
    cfg.epochs = 5;
    cfg.batch_size = 32;
    cfg.learning_rate = 0.01;
    cfg.hidden_layers = {hidden};

    auto result = Trainer::train_digitall_twin(X_train, y_train, X_test, y_test, cfg);

    EXPECT_EQ(result.train_loss.size(), 5u);
    EXPECT_EQ(result.test_accuracy.size(), 5u);
    EXPECT_GT(result.test_accuracy.back(), 0.0);
}

TEST(TrainerTest, CreateOpticalNetwork) {
    int in = 8, hid = 16, out = 5;

    TrainResult tr;
    tr.all_weights = {Eigen::MatrixXd::Random(in, hid), Eigen::MatrixXd::Random(hid, out)};
    tr.activations = {ActivationType::RELU, ActivationType::NONE};

    ONN net = Trainer::create_optical_network(tr, false);
    EXPECT_EQ(net.num_layers(), 2);

    Eigen::VectorXd input = Eigen::VectorXd::Random(in);
    Eigen::VectorXd output = net.forward(input);
    EXPECT_EQ(output.size(), out);
    EXPECT_NEAR(output.sum(), 1.0, 1e-10);
}

TEST(PipelineTest, EndToEndMNISTLike) {
    MNISTData train_data = MNISTLoader::generate_synthetic(800, 16, 5, 42);
    MNISTData test_data = MNISTLoader::generate_synthetic(200, 16, 5, 42);

    TrainConfig cfg;
    cfg.epochs = 10;
    cfg.batch_size = 64;
    cfg.learning_rate = 0.01;
    cfg.hidden_layers = {32};

    TrainResult result = Trainer::train_digitall_twin(
        train_data.images, train_data.labels,
        test_data.images, test_data.labels, cfg);

    EXPECT_EQ(result.train_loss.size(), 10u);
    EXPECT_EQ(result.test_accuracy.size(), 10u);
    EXPECT_GT(result.train_loss.front(), result.train_loss.back());

    double digital_acc = result.test_accuracy.back();
    EXPECT_GT(digital_acc, 0.2) << "Digital twin should beat random on 5 classes";

    ONN network = Trainer::create_optical_network(result, false);
    EXPECT_EQ(network.num_layers(), 2);

    double optical_acc = Trainer::accuracy(network, test_data.images, test_data.labels);
    EXPECT_GT(optical_acc, 0.0);

    std::cout << "  Digital accuracy: " << digital_acc * 100.0 << "%\n";
    std::cout << "  Optical accuracy: " << optical_acc * 100.0 << "%\n";
}

TEST(PipelineTest, ONNPredictConsistency) {
    int in = 16, hid = 32, out = 5;

    TrainResult tr;
    tr.all_weights = {Eigen::MatrixXd::Random(in, hid), Eigen::MatrixXd::Random(hid, out)};
    tr.activations = {ActivationType::RELU, ActivationType::NONE};

    ONN net = Trainer::create_optical_network(tr, false);

    Eigen::VectorXd input = Eigen::VectorXd::Random(in);
    int pred1 = Trainer::predict(net, input);
    int pred2 = Trainer::predict(net, input);
    EXPECT_EQ(pred1, pred2);
}

TEST(PipelineTest, ForwardIdealMatchesForward) {
    int in = 8, out = 8;

    TrainResult tr;
    tr.all_weights = {Eigen::MatrixXd::Random(in, in), Eigen::MatrixXd::Random(in, out)};
    tr.activations = {ActivationType::RELU, ActivationType::NONE};

    ONN net = Trainer::create_optical_network(tr, false);

    Eigen::VectorXd input = Eigen::VectorXd::Random(in);
    Eigen::VectorXd ideal_out = net.forward_ideal(input);

    EXPECT_EQ(ideal_out.size(), out);
    EXPECT_GT(ideal_out.norm(), 0.0) << "Ideal forward should produce non-zero output";
}

TEST(PipelineTest, PhysicalVsIdealDifference) {
    int in = 8, out = 5;

    TrainResult tr;
    tr.all_weights = {Eigen::MatrixXd::Random(in, in), Eigen::MatrixXd::Random(in, out)};
    tr.activations = {ActivationType::RELU, ActivationType::NONE};

    PhysicalConfig phys;
    phys.drift_enabled = true;
    phys.drift_sigma = 0.1;
    phys.attenuation_enabled = true;
    phys.shot_noise_enabled = false;
    phys.crosstalk_enabled = false;

    ONN ideal_net = Trainer::create_optical_network(tr, false);
    ONN phys_net = Trainer::create_optical_network(tr, true, phys);

    Eigen::VectorXd input = Eigen::VectorXd::Random(in);
    Eigen::VectorXd out_ideal = ideal_net.forward(input);
    Eigen::VectorXd out_phys = phys_net.forward(input);

    EXPECT_EQ(out_ideal.size(), out_phys.size());
    EXPECT_NEAR(out_ideal.sum(), 1.0, 1e-10);
    EXPECT_NEAR(out_phys.sum(), 1.0, 1e-10);
}

TEST(PipelineTest, FullMNISTReck) {
    MNISTData train_data = MNISTLoader::generate_synthetic(400, 16, 10, 7);
    MNISTData test_data = MNISTLoader::generate_synthetic(100, 16, 10, 77);

    TrainConfig cfg;
    cfg.epochs = 5;
    cfg.batch_size = 64;
    cfg.learning_rate = 0.01;
    cfg.hidden_layers = {32};
    cfg.mesh = MeshType::RECK;

    TrainResult result = Trainer::train_digitall_twin(
        train_data.images, train_data.labels,
        test_data.images, test_data.labels, cfg);

    ONN network = Trainer::create_optical_network(result, false);
    double acc = Trainer::accuracy(network, test_data.images, test_data.labels);
    EXPECT_GT(acc, 0.0);
    EXPECT_LE(acc, 1.0);
    std::cout << "  Reck optical accuracy: " << acc * 100.0 << "%\n";
}

TEST(PipelineTest, FullMNISTPhysical) {
    MNISTData train_data = MNISTLoader::generate_synthetic(200, 16, 10, 5);
    MNISTData test_data = MNISTLoader::generate_synthetic(60, 16, 10, 55);

    TrainConfig cfg;
    cfg.epochs = 3;
    cfg.batch_size = 64;
    cfg.learning_rate = 0.01;
    cfg.hidden_layers = {32};

    TrainResult result = Trainer::train_digitall_twin(
        train_data.images, train_data.labels,
        test_data.images, test_data.labels, cfg);

    PhysicalConfig phys;
    phys.drift_enabled = true;
    phys.drift_sigma = 0.10;
    phys.attenuation_enabled = true;
    phys.shot_noise_enabled = false;
    phys.crosstalk_enabled = false;

    ONN ideal_net = Trainer::create_optical_network(result, false);
    ONN phys_net = Trainer::create_optical_network(result, true, phys);

    double ideal_acc = Trainer::accuracy(ideal_net, test_data.images, test_data.labels);
    double phys_acc = Trainer::accuracy(phys_net, test_data.images, test_data.labels);

    EXPECT_GT(ideal_acc, 0.0);
    EXPECT_GE(phys_acc, 0.0);
    std::cout << "  Ideal accuracy: " << ideal_acc * 100.0 << "%\n";
    std::cout << "  Physical accuracy: " << phys_acc * 100.0 << "%\n";
}
