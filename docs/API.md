# API Reference

## Core Types

### `onn::Complex`
```cpp
using Complex = std::complex<double>;
```

### `onn::MeshType`
```cpp
enum class MeshType { RECK, CLEMENTS };
```

### `onn::ActivationType`
```cpp
enum class ActivationType { NONE, RELU, SIGMOID, SOFTMAX };
```

---

## MZI (`src/photonic/mzi.h`)

```cpp
class MZI {
public:
    MZI(double theta, double phi);
    
    // Static constructors
    static MZI bar_state();    // θ=0, φ=0 (pass-through)
    static MZI cross_state(); // θ=π, φ=π/2 (swap)
    
    // Apply to mode vector
    void apply(VectorXcd& modes, int i, int j) const;
    
    // Get transfer matrix
    Matrix2cd transfer_matrix() const;
    
    double theta() const;
    double phi() const;
};
```

**Example:**
```cpp
MZI mzi(M_PI/4, 0);  // 50/50 splitter
VectorXcd modes(2);
modes << Complex{1,0}, Complex{0,0};
mzi.apply(modes, 0, 1);
// modes[0] ≈ 0.707, modes[1] ≈ 0.707
```

---

## Reck Decomposition (`src/photonic/reck.h`)

```cpp
class ReckDecomposer {
public:
    static std::vector<MZIRecord> decompose(const MatrixXcd& U);
    static MatrixXcd reconstruct(const std::vector<MZIRecord>& mzis, int N);
    static double error(const MatrixXcd& U, const std::vector<MZIRecord>& mzis);
};
```

**Example:**
```cpp
MatrixXcd U = MatrixXcd::Random(8, 8);
auto mzis = ReckDecomposer::decompose(U);
double err = ReckDecomposer::error(U, mzis);  // < 1e-8
```

---

## Clements Decomposition (`src/photonic/clements.h`)

```cpp
class ClementsDecomposer {
public:
    static std::vector<MZIRecord> decompose(const MatrixXcd& U);
    static MatrixXcd reconstruct(const std::vector<MZIRecord>& mzis, int N);
    static double error(const MatrixXcd& U, const std::vector<MZIRecord>& mzis);
};
```

---

## SVD Mapper (`src/photonic/svd_mapper.h`)

```cpp
class SVDMapper {
public:
    static SVDResult decompose(const MatrixXcd& W);
    static OpticalMapping map_to_optical(const MatrixXcd& W, MeshType mesh);
    static VectorXcd forward(const OpticalMapping& map, const VectorXcd& input);
    static VectorXcd forward_with_loss(const OpticalMapping& map, 
                                       const VectorXcd& input, 
                                       double loss_per_mzi_db);
};
```

**Example:**
```cpp
MatrixXcd W = MatrixXcd::Random(256, 784);
auto mapping = SVDMapper::map_to_optical(W, MeshType::CLEMENTS);
VectorXcd input = VectorXcd::Random(784);
auto output = SVDMapper::forward(mapping, input);
// output ≈ W * input
```

---

## Physical Effects (`src/physical/physical_layer.h`)

### PhysicalConfig
```cpp
struct PhysicalConfig {
    bool drift_enabled = true;
    double drift_sigma = 0.10;           // Phase drift std dev
    
    bool attenuation_enabled = true;
    double attenuation_db_per_cm = 0.3;  // Propagation loss
    double coupling_loss_db = 0.1;       // Coupling loss per facet
    
    bool shot_noise_enabled = false;
    double quantum_efficiency = 0.9;     // Detector QE
    
    bool crosstalk_enabled = false;
    double crosstalk_kappa = 0.05;       // Coupling strength
    double crosstalk_sigma = 2.0;        // Spatial decay
    
    double time_step = 1.0;              // Calibration interval (s)
    unsigned seed = 42;                  // RNG seed
};
```

### PhysicalLayer
```cpp
class PhysicalLayer {
public:
    PhysicalLayer(const PhysicalConfig& config);
    
    VectorXcd apply(VectorXcd modes,
                    const std::vector<MZIRecord>& mzis,
                    const std::vector<MZIPosition>& positions);
    
    void advance_time(double dt);
    void reset();
};
```

---

## Linear Layer (`src/network/linear_layer.h`)

```cpp
class LinearLayer {
public:
    LinearLayer(int in_features, int out_features,
                MeshType mesh, bool physical,
                const PhysicalConfig& phys_config);
    
    // Forward passes
    VectorXcd forward(const VectorXcd& input);      // Physical (with effects)
    VectorXcd forward_ideal(const VectorXcd& input); // Ideal (no effects)
    
    // Weight management
    const MatrixXcd& weight_matrix() const;
    void set_weight_matrix(const MatrixXcd& W);      // Sets weights + decomposes
    void set_weight_only(const MatrixXcd& W);        // Sets weights only (fast)
    
    // Mesh access (lazy computation)
    const OpticalMapping& optical_mapping() const;
    void ensure_mapping() const;
    
    // Drift management
    void reset_drift();
    void apply_drift_state();
    
    int in_features() const;
    int out_features() const;
};
```

---

## Network (`src/network/network.h`)

```cpp
class ONN {
public:
    ONN(const std::vector<LayerConfig>& configs, bool physical,
        const PhysicalConfig& phys_config);
    
    VectorXd forward(const VectorXd& input_real);       // Physical
    VectorXd forward_ideal(const VectorXd& input_real);  // Ideal
    
    void load_weights(int layer_idx, const MatrixXcd& W);
    void compensate(const MatrixXd& X_ref, const VectorXi& y_ref,
                    const CalibrationConfig& cal_config);
    
    int num_layers() const;
    const LinearLayer& layer(int i) const;
    LinearLayer& layer_mutable(int i);
};
```

**Example:**
```cpp
std::vector<LayerConfig> configs = {
    {784, 256, ActivationType::RELU, MeshType::CLEMENTS},
    {256, 128, ActivationType::RELU, MeshType::CLEMENTS},
    {128, 10, ActivationType::NONE, MeshType::CLEMENTS}
};

ONN network(configs, true, PhysicalConfig{});
network.load_weights(0, W1.cast<Complex>());
network.load_weights(1, W2.cast<Complex>());
network.load_weights(2, W3.cast<Complex>());

VectorXd output = network.forward(input);
int predicted;
output.maxCoeff(&predicted);
```

---

## Trainer (`src/network/trainer.h`)

### TrainConfig
```cpp
struct TrainConfig {
    int epochs = 10;
    int batch_size = 32;
    double learning_rate = 0.001;
    std::vector<int> hidden_layers = {256};
    MeshType mesh = MeshType::CLEMENTS;
};
```

### Trainer
```cpp
class Trainer {
public:
    static TrainResult train_digitall_twin(
        const MatrixXd& X_train, const VectorXi& y_train,
        const MatrixXd& X_test, const VectorXi& y_test,
        const TrainConfig& config);
    
    static ONN create_optical_network(
        const TrainResult& result, bool physical,
        const PhysicalConfig& phys_config);
    
    static int predict(ONN& network, const VectorXd& input);
    static int predict_ideal(ONN& network, const VectorXd& input);
    static double accuracy(ONN& network, const MatrixXd& X, const VectorXi& y);
    static double ideal_accuracy(ONN& network, const MatrixXd& X, const VectorXi& y);
};
```

**Example:**
```cpp
TrainConfig cfg;
cfg.epochs = 10;
cfg.batch_size = 128;
cfg.learning_rate = 0.01;
cfg.hidden_layers = {256, 128};

TrainResult result = Trainer::train_digitall_twin(
    X_train, y_train, X_test, y_test, cfg);

ONN network = Trainer::create_optical_network(result, true, PhysicalConfig{});
double acc = Trainer::accuracy(network, X_test, y_test);
std::cout << "Accuracy: " << acc * 100.0 << "%\n";
```

---

## MNIST Loader (`src/data/mnist_loader.h`)

```cpp
struct MNISTData {
    MatrixXd images;   // (N, 784)
    VectorXi labels;   // (N,)
    int num_samples;
    int image_size;
};

class MNISTLoader {
public:
    static MNISTData load(const std::string& dir);
    static MNISTData load_subset(const std::string& dir, int max_samples);
    static MNISTData generate_synthetic(int num_samples, int input_size,
                                        int num_classes, unsigned seed);
};
```
