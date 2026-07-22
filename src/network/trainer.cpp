#include "network/trainer.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

namespace onn {

static Eigen::MatrixXd relu_forward(const Eigen::MatrixXd& x) {
    return x.array().max(0.0);
}

static Eigen::MatrixXd relu_backward(const Eigen::MatrixXd& dOut, const Eigen::MatrixXd& x) {
    Eigen::ArrayXXd mask = (x.array() > 0.0)
        .select(Eigen::ArrayXXd::Ones(x.rows(), x.cols()),
                Eigen::ArrayXXd::Zero(x.rows(), x.cols()));
    return dOut.array() * mask;
}

static Eigen::MatrixXd softmax_forward(const Eigen::MatrixXd& x) {
    Eigen::MatrixXd centered = x;
    for (int i = 0; i < x.rows(); i++) {
        centered.row(i) -= Eigen::VectorXd::Constant(x.cols(), x.row(i).maxCoeff());
    }
    Eigen::MatrixXd exp_x = centered.array().exp();
    Eigen::VectorXd row_sums = exp_x.rowwise().sum();
    for (int i = 0; i < exp_x.rows(); i++) {
        exp_x.row(i) /= row_sums(i);
    }
    return exp_x;
}

static double cross_entropy_loss(const Eigen::MatrixXd& pred, const Eigen::VectorXi& labels) {
    double loss = 0.0;
    int batch = pred.rows();
    for (int i = 0; i < batch; i++) {
        int label = labels(i);
        loss -= std::log(std::max(pred(i, label), 1e-10));
    }
    return loss / batch;
}

struct AdamState {
    Eigen::MatrixXd mW;
    Eigen::MatrixXd vW;
    Eigen::VectorXd mb;
    Eigen::VectorXd vb;
};

TrainResult Trainer::train_digitall_twin(
    const Eigen::MatrixXd& X_train,
    const Eigen::VectorXi& y_train,
    const Eigen::MatrixXd& X_test,
    const Eigen::VectorXi& y_test,
    const TrainConfig& config) {

    int input_size = X_train.cols();
    int num_classes = static_cast<int>(y_train.maxCoeff()) + 1;
    int L = static_cast<int>(config.hidden_layers.size()) + 1;

    std::vector<int> layer_sizes;
    layer_sizes.push_back(input_size);
    for (int h : config.hidden_layers) {
        layer_sizes.push_back(h);
    }
    layer_sizes.push_back(num_classes);

    std::mt19937 rng(42);
    std::vector<Eigen::MatrixXd> W(L);
    std::vector<Eigen::VectorXd> b(L);
    for (int i = 0; i < L; i++) {
        int fan_in = layer_sizes[i];
        int fan_out = layer_sizes[i + 1];
        double stddev = std::sqrt(2.0 / fan_in);
        std::normal_distribution<double> dist(0.0, stddev);
        W[i] = Eigen::MatrixXd(fan_in, fan_out);
        for (int r = 0; r < fan_in; r++)
            for (int c = 0; c < fan_out; c++)
                W[i](r, c) = dist(rng);
        b[i] = Eigen::VectorXd::Zero(fan_out);
    }

    std::vector<AdamState> adam(L);
    for (int i = 0; i < L; i++) {
        adam[i].mW = Eigen::MatrixXd::Zero(W[i].rows(), W[i].cols());
        adam[i].vW = Eigen::MatrixXd::Zero(W[i].rows(), W[i].cols());
        adam[i].mb = Eigen::VectorXd::Zero(b[i].size());
        adam[i].vb = Eigen::VectorXd::Zero(b[i].size());
    }

    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double eps = 1e-8;
    double lr = config.learning_rate;
    int N = X_train.rows();
    int batch_size = std::min(config.batch_size, N);
    int total_batches = (N + batch_size - 1) / batch_size;
    int global_step = 0;

    TrainResult result;
    result.input_size = input_size;
    result.num_classes = num_classes;
    result.all_weights.resize(L);
    result.activations.resize(L, ActivationType::RELU);
    result.activations.back() = ActivationType::NONE;

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        double epoch_loss = 0.0;
        int batches = 0;

        double cosine_factor = 0.5 * (1.0 + std::cos(M_PI * epoch / config.epochs));
        double epoch_lr = lr * (0.1 + 0.9 * cosine_factor);

        std::vector<int> indices(N);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (int start = 0; start < N; start += batch_size) {
            int end = std::min(start + batch_size, N);
            int B = end - start;

            Eigen::MatrixXd x_batch(B, input_size);
            Eigen::VectorXi y_batch(B);
            for (int k = 0; k < B; k++) {
                x_batch.row(k) = X_train.row(indices[start + k]);
                y_batch(k) = y_train(indices[start + k]);
            }

            std::vector<Eigen::MatrixXd> A(L + 1);
            std::vector<Eigen::MatrixXd> Z(L);
            A[0] = x_batch;

            for (int i = 0; i < L; i++) {
                Z[i] = A[i] * W[i];
                Z[i].rowwise() += b[i].transpose();
                if (i < L - 1) {
                    A[i + 1] = relu_forward(Z[i]);
                } else {
                    A[i + 1] = softmax_forward(Z[i]);
                }
            }

            Eigen::MatrixXd probs = A[L];
            double loss = cross_entropy_loss(probs, y_batch);
            epoch_loss += loss;
            batches++;

            Eigen::MatrixXd dA = probs;
            for (int i = 0; i < B; i++) {
                dA(i, y_batch(i)) -= 1.0;
            }
            dA /= B;

            for (int i = L - 1; i >= 0; i--) {
                Eigen::MatrixXd dW = A[i].transpose() * dA;
                Eigen::VectorXd db = dA.colwise().sum();

                if (i > 0) {
                    Eigen::MatrixXd dA_prev = dA * W[i].transpose();
                    dA = relu_backward(dA_prev, Z[i - 1]);
                }

                global_step++;
                adam[i].mW = beta1 * adam[i].mW + (1.0 - beta1) * dW;
                adam[i].vW = beta2 * adam[i].vW + (1.0 - beta2) * dW.array().square().matrix();
                adam[i].mb = beta1 * adam[i].mb + (1.0 - beta1) * db;
                adam[i].vb = beta2 * adam[i].vb + (1.0 - beta2) * db.array().square().matrix();

                double bc1 = 1.0 - std::pow(beta1, global_step);
                double bc2 = 1.0 - std::pow(beta2, global_step);

                Eigen::MatrixXd mW_hat = adam[i].mW / bc1;
                Eigen::MatrixXd vW_hat = adam[i].vW / bc2;
                Eigen::VectorXd mb_hat = adam[i].mb / bc1;
                Eigen::VectorXd vb_hat = adam[i].vb / bc2;

                W[i].array() -= epoch_lr * mW_hat.array() / (vW_hat.array().sqrt() + eps);
                b[i].array() -= epoch_lr * mb_hat.array() / (vb_hat.array().sqrt() + eps);
            }
        }

        result.train_loss.push_back(epoch_loss / batches);

        int correct = 0;
        int test_N = X_test.rows();
        for (int i = 0; i < test_N; i++) {
            Eigen::MatrixXd x = X_test.row(i);
            for (int j = 0; j < L; j++) {
                Eigen::MatrixXd z = x * W[j];
                z.rowwise() += b[j].transpose();
                if (j < L - 1) {
                    x = relu_forward(z);
                } else {
                    x = softmax_forward(z);
                }
            }
            int pred;
            x.row(0).maxCoeff(&pred);
            if (pred == y_test(i)) correct++;
        }
        result.test_accuracy.push_back(static_cast<double>(correct) / test_N);
    }

    result.all_weights.resize(L);
    for (int i = 0; i < L; i++) {
        result.all_weights[i] = W[i];
    }

    return result;
}

ONN Trainer::create_optical_network(
    const TrainResult& result,
    bool physical,
    const PhysicalConfig& phys_config) {

    std::vector<ActivationType> acts = result.activations;
    if (acts.empty()) {
        acts.resize(result.all_weights.size(), ActivationType::RELU);
        acts.back() = ActivationType::NONE;
    }
    return create_optical_network(result, acts, physical, phys_config);
}

ONN Trainer::create_optical_network(
    const TrainResult& result,
    const std::vector<ActivationType>& activations,
    bool physical,
    const PhysicalConfig& phys_config) {

    int L = static_cast<int>(result.all_weights.size());
    std::vector<LayerConfig> configs;

    for (int i = 0; i < L; i++) {
        int in_f = result.all_weights[i].rows();
        int out_f = result.all_weights[i].cols();
        ActivationType act = (i < static_cast<int>(activations.size()))
                            ? activations[i] : ActivationType::NONE;
        configs.push_back({in_f, out_f, act, MeshType::CLEMENTS});
    }

    ONN network(configs, physical, phys_config);

    for (int i = 0; i < L; i++) {
        network.load_weights(i, result.all_weights[i].transpose().cast<Complex>());
    }

    return network;
}

int Trainer::predict(ONN& network, const Eigen::VectorXd& input) {
    Eigen::VectorXd output = network.forward(input);
    int pred;
    output.maxCoeff(&pred);
    return pred;
}

int Trainer::predict_ideal(ONN& network, const Eigen::VectorXd& input) {
    Eigen::VectorXd output = network.forward_ideal(input);
    int pred;
    output.maxCoeff(&pred);
    return pred;
}

double Trainer::accuracy(ONN& network, const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    int correct = 0;
    for (int i = 0; i < X.rows(); i++) {
        int pred = predict(network, X.row(i));
        if (pred == y(i)) correct++;
    }
    return static_cast<double>(correct) / X.rows();
}

double Trainer::ideal_accuracy(ONN& network, const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    int correct = 0;
    for (int i = 0; i < X.rows(); i++) {
        int pred = predict_ideal(network, X.row(i));
        if (pred == y(i)) correct++;
    }
    return static_cast<double>(correct) / X.rows();
}

} // namespace onn
