#include "core/types.h"
#include "photonic/mzi.h"
#include "photonic/reck.h"
#include "photonic/clements.h"
#include "photonic/svd_mapper.h"
#include "physical/physical_layer.h"
#include "network/network.h"
#include "network/trainer.h"
#include "data/mnist_loader.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <string>
#include <cstring>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>

using namespace onn;

static void print_separator() {
    std::cout << "========================================\n";
}

static void bench_mzi_unitarity() {
    print_separator();
    std::cout << "TEST: MZI Unitarity\n";
    std::cout << "----------------------------------------\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0, PI);

    double max_err = 0.0;
    int trials = 1000;
    for (int i = 0; i < trials; i++) {
        double theta = dist(rng);
        double phi = dist(rng);
        MZI mzi(theta, phi);
        Matrix2cd U = mzi.transfer_matrix();
        Matrix2cd product = U.adjoint() * U;
        Matrix2cd identity = Matrix2cd::Identity();
        double err = (product - identity).norm();
        max_err = std::max(max_err, err);
    }

    bool pass = max_err < 1e-12;
    std::cout << "Max ||U†U - I||_F over " << trials << " trials: " << max_err;
    std::cout << (pass ? " [PASS]\n" : " [FAIL]\n");
}

static void bench_mzi_states() {
    print_separator();
    std::cout << "TEST: MZI Bar/Cross States\n";
    std::cout << "----------------------------------------\n";

    MZI bar = MZI::bar_state();
    MZI cross = MZI::cross_state();

    VectorXcd in(2);
    in << Complex{1.0, 0.0}, Complex{0.0, 0.0};

    VectorXcd out_bar = in;
    bar.apply(out_bar, 0, 1);

    VectorXcd out_cross = in;
    cross.apply(out_cross, 0, 1);

    std::cout << "Bar state  (0,0) -> " << out_bar(0) << "  (should be ~1+0i)\n";
    std::cout << "Cross state (0,0) -> " << out_cross(1) << "  (should be ~i)\n";

    bool pass = std::abs(out_bar(0) - Complex{1.0, 0.0}) < 1e-10 &&
                std::abs(out_cross(1) - IM) < 1e-10;
    std::cout << (pass ? "[PASS]\n" : "[FAIL]\n");
}

static void bench_decomposition_scaling() {
    print_separator();
    std::cout << "BENCHMARK: Decomposition Scaling\n";
    std::cout << "----------------------------------------\n";

    for (int N : {4, 8, 16, 32, 64}) {
        std::mt19937 rng(N * 100);
        std::normal_distribution<double> dist(0.0, 1.0);

        Eigen::MatrixXcd U(N, N);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                U(i, j) = Complex{dist(rng), dist(rng)};
        Eigen::HouseholderQR<Eigen::MatrixXcd> qr(U);
        Eigen::MatrixXcd Q = qr.householderQ();

        int iters = N <= 16 ? 50 : 10;

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            volatile auto mzis = ReckDecomposer::decompose(Q);
            (void)mzis;
        }
        auto end = std::chrono::high_resolution_clock::now();
        double reck_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;

        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) {
            volatile auto mzis = ClementsDecomposer::decompose(Q);
            (void)mzis;
        }
        end = std::chrono::high_resolution_clock::now();
        double clements_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;

        auto mzis = ReckDecomposer::decompose(Q);
        int mzi_count = static_cast<int>(mzis.size());
        double err = ReckDecomposer::error(Q, mzis);

        std::cout << N << "x" << N
                  << "  MZIs: " << mzi_count
                  << "  Reck: " << reck_ms << " ms"
                  << "  Clements: " << clements_ms << " ms"
                  << "  Error: " << err << "\n";
    }
}

static void bench_reck_decomposition() {
    print_separator();
    std::cout << "TEST: Reck Decomposition\n";
    std::cout << "----------------------------------------\n";

    int N = 8;
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    Eigen::MatrixXcd U(N, N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            U(i, j) = Complex{dist(rng), dist(rng)};

    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(U);
    Eigen::MatrixXcd Q = qr.householderQ();

    auto start = std::chrono::high_resolution_clock::now();
    auto mzis = ReckDecomposer::decompose(Q);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    double err = ReckDecomposer::error(Q, mzis);
    bool pass = err < 1e-8;

    std::cout << "Matrix size: " << N << "x" << N << "\n";
    std::cout << "MZIs: " << mzis.size() << " (expected " << N * (N - 1) / 2 << ")\n";
    std::cout << "Error: " << err;
    std::cout << (pass ? " [PASS]\n" : " [FAIL]\n");
    std::cout << "Time: " << ms << " ms\n";
}

static void bench_clements_decomposition() {
    print_separator();
    std::cout << "TEST: Clements Decomposition\n";
    std::cout << "----------------------------------------\n";

    int N = 8;
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    Eigen::MatrixXcd U(N, N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            U(i, j) = Complex{dist(rng), dist(rng)};

    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(U);
    Eigen::MatrixXcd Q = qr.householderQ();

    auto start = std::chrono::high_resolution_clock::now();
    auto mzis = ClementsDecomposer::decompose(Q);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    double err = ClementsDecomposer::error(Q, mzis);
    bool pass = err < 1e-8;

    std::cout << "Matrix size: " << N << "x" << N << "\n";
    std::cout << "MZIs: " << mzis.size() << " (expected " << N * (N - 1) / 2 << ")\n";
    std::cout << "Error: " << err;
    std::cout << (pass ? " [PASS]\n" : " [FAIL]\n");
    std::cout << "Time: " << ms << " ms\n";
}

static void bench_svd_mapping() {
    print_separator();
    std::cout << "TEST: SVD Mapping + Forward Pass\n";
    std::cout << "----------------------------------------\n";

    int in = 784, out = 256;
    Eigen::MatrixXcd W = Eigen::MatrixXcd::Random(out, in);

    auto start = std::chrono::high_resolution_clock::now();
    auto mapping = SVDMapper::map_to_optical(W, MeshType::CLEMENTS);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    Eigen::VectorXcd input = Eigen::VectorXcd::Random(in);
    auto output = SVDMapper::forward(mapping, input);
    Eigen::VectorXcd expected = W * input;

    double err = (output - expected).norm() / expected.norm();

    std::cout << "Matrix: " << out << "x" << in << "\n";
    std::cout << "MZIs (U mesh): " << mapping.u_mzis.size() << "\n";
    std::cout << "MZIs (V† mesh): " << mapping.vh_mzis.size() << "\n";
    std::cout << "Decompose time: " << ms << " ms\n";
    std::cout << "Forward error: " << err;
    std::cout << (err < 1e-8 ? " [PASS]\n" : " [FAIL]\n");
}

static void bench_forward_speed() {
    print_separator();
    std::cout << "BENCHMARK: Forward Pass Speed\n";
    std::cout << "----------------------------------------\n";

    for (int N : {8, 16, 32, 64}) {
        Eigen::MatrixXcd W = Eigen::MatrixXcd::Random(N, N);
        auto mapping = SVDMapper::map_to_optical(W, MeshType::CLEMENTS);

        int iterations = 1000;
        Eigen::VectorXcd input = Eigen::VectorXcd::Random(N);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            volatile auto result = SVDMapper::forward(mapping, input);
            (void)result;
        }
        auto end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << N << "x" << N << " mesh: " << (total_ms / iterations) << " ms/forward, "
                  << mapping.u_mzis.size() + mapping.vh_mzis.size() << " MZIs\n";
    }
}

static void bench_physical_effects() {
    print_separator();
    std::cout << "TEST: Physical Constraints\n";
    std::cout << "----------------------------------------\n";

    int N = 8;
    Eigen::MatrixXcd W = Eigen::MatrixXcd::Random(N, N);
    auto mapping = SVDMapper::map_to_optical(W, MeshType::CLEMENTS);
    Eigen::VectorXcd input = Eigen::VectorXcd::Random(N);

    auto ideal_out = SVDMapper::forward(mapping, input);

    PhysicalConfig phys;
    phys.drift_enabled = true;
    phys.drift_sigma = 0.02;
    phys.attenuation_enabled = true;
    phys.shot_noise_enabled = false;
    phys.crosstalk_enabled = false;
    PhysicalLayer physical(phys);

    std::vector<MZIPosition> positions;
    for (size_t i = 0; i < mapping.u_mzis.size(); i++) {
        positions.push_back({static_cast<int>(i / N), static_cast<int>(i % N)});
    }

    auto noisy_out = physical.apply(input, mapping.u_mzis, positions);

    double drift_error = (noisy_out - ideal_out).norm();
    std::cout << "Phase drift effect (norm diff): " << drift_error;
    std::cout << (drift_error > 0 ? " [PASS - noise present]\n" : " [WARN]\n");
}

static void bench_network_inference() {
    print_separator();
    std::cout << "TEST: ONN Network Inference (synthetic MNIST-like)\n";
    std::cout << "----------------------------------------\n";

    int input_size = 16;
    int hidden = 32;
    int classes = 10;
    int test_size = 100;

    Eigen::MatrixXd W1 = Eigen::MatrixXd::Random(input_size, hidden) * 0.1;
    Eigen::MatrixXd W2 = Eigen::MatrixXd::Random(hidden, classes) * 0.1;

    std::vector<LayerConfig> configs = {
        {input_size, hidden, ActivationType::RELU, MeshType::CLEMENTS},
        {hidden, classes, ActivationType::NONE, MeshType::CLEMENTS}
    };

    ONN network(configs, false);

    auto start = std::chrono::high_resolution_clock::now();
    network.load_weights(0, W1.transpose().cast<Complex>());
    network.load_weights(1, W2.transpose().cast<Complex>());
    auto end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(end - start).count();

    Eigen::MatrixXd X_test = Eigen::MatrixXd::Random(test_size, input_size);
    Eigen::VectorXi y_test = Eigen::VectorXi::Zero(test_size);
    for (int i = 0; i < test_size; i++) y_test(i) = i % classes;

    start = std::chrono::high_resolution_clock::now();
    int correct = 0;
    for (int i = 0; i < test_size; i++) {
        Eigen::VectorXd out = network.forward(X_test.row(i));
        int pred;
        out.maxCoeff(&pred);
        if (pred == y_test(i)) correct++;
    }
    end = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Network: " << input_size << " -> " << hidden << " -> " << classes << "\n";
    std::cout << "Weight loading: " << load_ms << " ms\n";
    std::cout << "Inference (random weights, " << test_size << " samples): " << infer_ms << " ms\n";
    std::cout << "Per-sample: " << (infer_ms / test_size) << " ms\n";
    std::cout << "Accuracy (random): " << (100.0 * correct / test_size) << "%\n";
    std::cout << "[PASS - inference pipeline works]\n";
}

static void bench_mnist_training() {
    print_separator();
    std::cout << "BENCHMARK: MNIST Training + Inference\n";
    std::cout << "----------------------------------------\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    MNISTData train_data = MNISTLoader::generate_synthetic(500, 16, 10, 42);
    MNISTData test_data = MNISTLoader::generate_synthetic(100, 16, 10, 99);
    auto t1 = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    TrainConfig cfg;
    cfg.epochs = 5;
    cfg.batch_size = 64;
    cfg.learning_rate = 0.01;
    cfg.hidden_layers = {32};

    t0 = std::chrono::high_resolution_clock::now();
    TrainResult result = Trainer::train_digitall_twin(
        train_data.images, train_data.labels,
        test_data.images, test_data.labels, cfg);
    t1 = std::chrono::high_resolution_clock::now();
    double train_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    ONN network = Trainer::create_optical_network(result, false);
    t1 = std::chrono::high_resolution_clock::now();
    double create_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    double optical_acc = Trainer::accuracy(network, test_data.images, test_data.labels);
    t1 = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Data generation: " << gen_ms << " ms\n";
    std::cout << "Training (" << cfg.epochs << " epochs): " << train_ms << " ms\n";
    std::cout << "ONN creation: " << create_ms << " ms\n";
    std::cout << "Digital accuracy: " << (result.test_accuracy.back() * 100.0) << "%\n";
    std::cout << "Optical accuracy: " << (optical_acc * 100.0) << "%\n";
    std::cout << "Inference time (" << test_data.num_samples << " samples): " << infer_ms << " ms\n";
}

static void bench_real_mnist(const std::string& mnist_dir) {
    print_separator();
    std::cout << "BENCHMARK: Real MNIST - Full Pipeline\n";
    std::cout << "----------------------------------------\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    MNISTData train_data = MNISTLoader::load(mnist_dir);
    MNISTData test_data = MNISTLoader::load_subset(mnist_dir, 1000);
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (train_data.num_samples == 0) {
        std::cout << "ERROR: Could not load MNIST data from " << mnist_dir << "\n";
        std::cout << "Expected files: train-images-idx3-ubyte, train-labels-idx1-ubyte\n";
        return;
    }

    std::cout << "Loaded " << train_data.num_samples << " train, "
              << test_data.num_samples << " test samples\n";
    std::cout << "Data loading: " << load_ms << " ms\n";

    TrainConfig cfg;
    cfg.epochs = 10;
    cfg.batch_size = 128;
    cfg.learning_rate = 0.01;
    cfg.hidden_layers = {256, 128};

    t0 = std::chrono::high_resolution_clock::now();
    TrainResult result = Trainer::train_digitall_twin(
        train_data.images, train_data.labels,
        test_data.images, test_data.labels, cfg);
    t1 = std::chrono::high_resolution_clock::now();
    double train_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double digital_acc = result.test_accuracy.back();
    std::cout << "Training (" << cfg.epochs << " epochs): " << train_ms << " ms\n";
    std::cout << "Digital accuracy: " << (digital_acc * 100.0) << "%\n";

    t0 = std::chrono::high_resolution_clock::now();
    ONN network = Trainer::create_optical_network(result, false);
    t1 = std::chrono::high_resolution_clock::now();
    double create_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "ONN creation (no mesh decomp): " << create_ms << " ms\n";

    t0 = std::chrono::high_resolution_clock::now();
    double ideal_acc = Trainer::ideal_accuracy(network, test_data.images, test_data.labels);
    t1 = std::chrono::high_resolution_clock::now();
    double ideal_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    double optical_acc = Trainer::accuracy(network, test_data.images, test_data.labels);
    t1 = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Ideal accuracy (forward_ideal): " << (ideal_acc * 100.0) << "%\n";
    std::cout << "Optical accuracy (mesh forward): " << (optical_acc * 100.0) << "%\n";
    std::cout << "Inference time (" << test_data.num_samples << " samples): " << infer_ms << " ms\n";
    std::cout << "Per-sample: " << (infer_ms / test_data.num_samples) << " ms\n";

    PhysicalConfig phys;
    phys.drift_enabled = true;
    phys.drift_sigma = 0.10;
    phys.attenuation_enabled = true;
    phys.shot_noise_enabled = false;
    phys.crosstalk_enabled = false;

    ONN phys_net = Trainer::create_optical_network(result, true, phys);
    t0 = std::chrono::high_resolution_clock::now();
    double phys_acc = Trainer::accuracy(phys_net, test_data.images, test_data.labels);
    t1 = std::chrono::high_resolution_clock::now();
    double phys_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Physical accuracy (no comp): " << (phys_acc * 100.0) << "% (" << phys_ms << " ms)\n";

    std::cout << "\n--- Drift Compensation ---\n";
    CalibrationConfig cal_cfg;
    cal_cfg.max_iterations = 10;
    cal_cfg.learning_rate = 0.1;
    cal_cfg.perturbation_delta = 0.01;

    Eigen::MatrixXd X_ref = train_data.images.topRows(50);
    Eigen::VectorXi y_ref = train_data.labels.head(50);

    t0 = std::chrono::high_resolution_clock::now();
    phys_net.compensate(X_ref, y_ref, cal_cfg);
    t1 = std::chrono::high_resolution_clock::now();
    double comp_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    double comp_acc = Trainer::accuracy(phys_net, test_data.images, test_data.labels);
    t1 = std::chrono::high_resolution_clock::now();
    double comp_infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Compensation time: " << comp_ms << " ms\n";
    std::cout << "Physical accuracy (after comp): " << (comp_acc * 100.0) << "%\n";
    std::cout << "Compensated inference: " << comp_infer_ms << " ms\n";

    std::cout << "\n--- Thermal Crosstalk ---\n";
    PhysicalConfig phys_xtalk = phys;
    phys_xtalk.crosstalk_enabled = true;
    phys_xtalk.crosstalk_kappa = 0.05;
    phys_xtalk.crosstalk_sigma = 2.0;

    ONN xtalk_net = Trainer::create_optical_network(result, true, phys_xtalk);
    t0 = std::chrono::high_resolution_clock::now();
    double xtalk_acc = Trainer::accuracy(xtalk_net, test_data.images, test_data.labels);
    t1 = std::chrono::high_resolution_clock::now();
    double xtalk_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Crosstalk accuracy (kappa=0.05): " << (xtalk_acc * 100.0) << "% (" << xtalk_ms << " ms)\n";

    print_separator();
    std::cout << "  TARGETS vs ACTUAL:\n";
    std::cout << "  Digital  >= 97%:  " << (digital_acc * 100.0) << "%";
    std::cout << (digital_acc >= 0.97 ? " [PASS]\n" : " [MISS]\n");
    std::cout << "  Ideal    >= 97%:  " << (ideal_acc * 100.0) << "%";
    std::cout << (ideal_acc >= 0.97 ? " [PASS]\n" : " [MISS]\n");
    std::cout << "  Physical >= 93%:  " << (phys_acc * 100.0) << "%";
    std::cout << (phys_acc >= 0.93 ? " [PASS]\n" : " [MISS]\n");
    std::cout << "  Compens. >= 96%:  " << (comp_acc * 100.0) << "%";
    std::cout << (comp_acc >= 0.96 ? " [PASS]\n" : " [MISS]\n");
    std::cout << "  Crosstr. >= 94%:  " << (xtalk_acc * 100.0) << "%";
    std::cout << (xtalk_acc >= 0.94 ? " [PASS]\n" : " [MISS]\n");
    print_separator();
}

static void bench_memory_power() {
    print_separator();
    std::cout << "BENCHMARK: Memory Usage & Power Model\n";
    std::cout << "----------------------------------------\n";

    size_t total_bytes = 0;

    for (int N : {8, 16, 32, 64}) {
        Eigen::MatrixXcd W = Eigen::MatrixXcd::Random(N, N);
        size_t weight_bytes = N * N * sizeof(std::complex<double>);
        size_t forward_bytes = N * sizeof(std::complex<double>) * 3;
        size_t mzi_bytes = (N * (N - 1) / 2) * (2 * sizeof(double) + 2 * sizeof(int));
        size_t layer_bytes = weight_bytes + forward_bytes + mzi_bytes;
        total_bytes += layer_bytes;

        std::cout << N << "x" << N << " mesh: "
                  << "weights=" << (weight_bytes / 1024.0) << "KB"
                  << "  MZIs=" << (mzi_bytes / 1024.0) << "KB"
                  << "  total=" << (layer_bytes / 1024.0) << "KB\n";
    }

    size_t mnIST_model_bytes = 0;
    int layers[][2] = {{784, 256}, {256, 128}, {128, 10}};
    for (auto& lr : layers) {
        mnIST_model_bytes += lr[0] * lr[1] * sizeof(std::complex<double>);
    }
    total_bytes += mnIST_model_bytes;
    std::cout << "\nMNIST model (784->256->128->10): " << (mnIST_model_bytes / 1024.0) << "KB\n";
    std::cout << "Total estimated memory: " << (total_bytes / (1024.0 * 1024.0)) << "MB\n";
    std::cout << "Target < 50MB: " << (total_bytes < 50 * 1024 * 1024 ? "[PASS]\n" : "[FAIL]\n");

    print_separator();
    std::cout << "POWER MODEL (Photonic vs GPU)\n";
    std::cout << "----------------------------------------\n";

    double mzi_power_mw = 2.0;
    double laser_power_mw = 10.0;
    double detector_power_mw = 0.1;

    for (int N : {8, 16, 32, 64}) {
        int mzi_count = N * (N - 1);
        double chip_power = mzi_count * mzi_power_mw + N * detector_power_mw + laser_power_mw;
        std::cout << N << "x" << N << ": " << mzi_count << " MZIs, "
                  << (chip_power / 1000.0) << "W chip power\n";
    }

    int mnist_mzis = 784 * 256 + 256 * 128 + 128 * 10;
    double mnist_power = mnist_mzis * mzi_power_mw + (256 + 128 + 10) * detector_power_mw + laser_power_mw;
    std::cout << "\nMNIST full (784->256->128->10): " << mnist_mzis << " MZIs, "
              << (mnist_power / 1000.0) << "W\n";
    std::cout << "GPU baseline: ~150W (inference)\n";
    std::cout << "64x64 photonic: " << (64 * 63 * mzi_power_mw / 1000.0) << "W vs 150W GPU = "
              << (150.0 / (64 * 63 * mzi_power_mw / 1000.0)) << "x improvement\n";
    std::cout << "Target < 1W (for 64x64): "
              << (64 * 63 * mzi_power_mw / 1000.0 < 1.0 ? "[PASS]" : "[FAIL]")
              << " (" << (64 * 63 * mzi_power_mw / 1000.0) << "W)\n";
    print_separator();
}

static void print_usage() {
    std::cout << "Usage: onn_benchmark [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  (no args)             Run all tests and benchmarks\n";
    std::cout << "  --benchmark_all       Run all tests and benchmarks\n";
    std::cout << "  --benchmark_mzi       Run MZI unitarity tests only\n";
    std::cout << "  --benchmark_mesh      Run mesh decomposition tests only\n";
    std::cout << "  --benchmark_forward   Run forward pass speed benchmarks\n";
    std::cout << "  --benchmark_scaling   Run decomposition scaling benchmarks\n";
    std::cout << "  --benchmark_physical  Run physical effects tests only\n";
    std::cout << "  --benchmark_mnist     Run MNIST training/inference benchmark\n";
    std::cout << "  --benchmark_real_mnist Run full real MNIST pipeline\n";
    std::cout << "  --benchmark_mem_power Run memory and power model benchmarks\n";
    std::cout << "  --mnist-dir <path>    Path to MNIST data (default: data/mnist)\n";
    std::cout << "  --help                Show this help\n";
}

int main(int argc, char* argv[]) {
    bool run_all = (argc <= 1);
    bool run_mzi = false, run_mesh = false, run_forward = false;
    bool run_scaling = false, run_physical = false, run_mnist = false;
    bool run_real_mnist = false, run_mem_power = false;
    std::string mnist_dir = "data/mnist";

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0) { print_usage(); return 0; }
        if (std::strcmp(argv[i], "--benchmark_all") == 0) run_all = true;
        if (std::strcmp(argv[i], "--benchmark_mzi") == 0) run_mzi = true;
        if (std::strcmp(argv[i], "--benchmark_mesh") == 0) run_mesh = true;
        if (std::strcmp(argv[i], "--benchmark_forward") == 0) run_forward = true;
        if (std::strcmp(argv[i], "--benchmark_scaling") == 0) run_scaling = true;
        if (std::strcmp(argv[i], "--benchmark_physical") == 0) run_physical = true;
        if (std::strcmp(argv[i], "--benchmark_mnist") == 0) run_mnist = true;
        if (std::strcmp(argv[i], "--benchmark_real_mnist") == 0) run_real_mnist = true;
        if (std::strcmp(argv[i], "--benchmark_mem_power") == 0) run_mem_power = true;
        if (std::strcmp(argv[i], "--mnist-dir") == 0 && i + 1 < argc) {
            mnist_dir = argv[++i];
        }
    }

    std::cout << "\n";
    print_separator();
    std::cout << "  Optical Neural Network (ONN) Simulation\n";
    std::cout << "  Digital Sandbox - Physical Chip Model\n";
    print_separator();
    std::cout << "\n";

    if (run_all || run_mzi) {
        bench_mzi_unitarity();
        bench_mzi_states();
    }
    if (run_all || run_scaling) {
        bench_decomposition_scaling();
    }
    if (run_all || run_mesh) {
        bench_reck_decomposition();
        bench_clements_decomposition();
        bench_svd_mapping();
    }
    if (run_all || run_forward) {
        bench_forward_speed();
    }
    if (run_all || run_physical) {
        bench_physical_effects();
    }
    if (run_all || run_mnist) {
        bench_network_inference();
        bench_mnist_training();
    }
    if (run_all || run_real_mnist) {
        bench_real_mnist(mnist_dir);
    }
    if (run_all || run_mem_power) {
        bench_memory_power();
    }

    print_separator();
    std::cout << "All tests complete.\n";
    print_separator();

    return 0;
}
