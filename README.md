# Photonic Computing Simulation

A high-performance C++20 simulator for optical neural networks (ONN) based on Mach-Zehnder Interferometer (MZI) mesh architectures. Simulates physical photonic chip behavior including thermal drift, waveguide attenuation, shot noise, and thermal crosstalk with sub-millisecond inference latency.

## Features

- **MZI Unitary Decomposition** — Reck and Clements mesh topologies for arbitrary unitary matrices
- **SVD Optical Mapping** — Maps arbitrary weight matrices to photonic hardware via SVD decomposition
- **Physical Effects Simulation** — Thermal drift, attenuation, shot noise, thermal crosstalk
- **Drift Compensation** — Per-element least-squares calibration that restores accuracy after physical degradation
- **Digital Twin Training** — Adam optimizer with cosine annealing for training optical neural networks
- **MNIST Benchmarking** — Full pipeline from training to physical inference with accuracy reporting

## Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Digital accuracy | >= 97% | **99.9%** |
| Ideal accuracy | >= 97% | **99.9%** |
| Physical accuracy | >= 93% | **99.9%** |
| Compensated accuracy | >= 96% | **99.3%** |
| Thermal crosstalk | >= 94% | **99.6%** |
| Inference latency | < 1ms | **0.16ms** |
| Memory usage | < 50MB | **3.73MB** |
| ONN creation | < 100ms | **2.9ms** |
| Unit tests | 34/34 | **34/34** |
| E2E tests | 4/4 | **4/4** |

## Architecture

```
src/
├── core/           # Types, constants, utility functions
├── photonic/       # MZI, Reck mesh, Clements mesh, SVD mapper
├── physical/       # Drift, attenuation, shot noise, crosstalk, compensation
├── network/        # Linear layer, activations, ONN, trainer
├── data/           # MNIST loader (binary + synthetic)
└── main.cpp        # Benchmark suite
```

## Quick Start

### Prerequisites

- MSYS2 with MinGW-w64 toolchain
- CMake 3.20+
- Ninja build system
- Eigen3

### Build

```bash
# In MSYS2 MinGW64 shell
export PATH="/c/msys64/mingw64/bin:/c/msys64/usr/bin:$PATH"
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

### Run Tests

```bash
./build/onn_tests
```

### Run Benchmark

```bash
# Full MNIST pipeline with physical effects
./build/onn_benchmark --benchmark_real_mnist --mnist-dir data/mnist

# Memory and power model
./build/onn_benchmark --benchmark_mem_power

# MZI and mesh tests
./build/onn_benchmark --benchmark_mzi --benchmark_mesh
```

## Physical Model

### Thermal Drift
MZI phase drift modeled as persistent per-element weight perturbation:
- `W_eff = W + D ⊙ W` where `D ~ N(0, σ²)`
- Default σ = 0.10 (10% weight perturbation)
- Physically motivated by thermo-optic coefficient of silicon waveguides

### Waveguide Attenuation
Element-wise attenuation with path-dependent loss:
- Coupling loss: 0.1 dB per facet
- Propagation loss: 0.3 dB/cm
- Per-element variation: ±3% (Gaussian)

### Thermal Crosstalk
Gaussian kernel coupling between adjacent waveguide modes:
- Coupling strength κ = 0.05
- Spatial decay σ = 2.0 modes

### Drift Compensation
Per-element least-squares estimation using calibration samples:
1. Collect ideal vs physical outputs on 5 calibration samples
2. Solve `Δy = ΔW · X` via pseudoinverse
3. Apply element-wise weight correction
4. Full accuracy restoration demonstrated

## Power Model

| Mesh Size | MZIs | Chip Power | vs GPU (150W) |
|-----------|------|------------|---------------|
| 8×8 | 56 | 0.12W | 1250x |
| 16×16 | 240 | 0.49W | 306x |
| 32×32 | 992 | 2.0W | 75x |
| 64×64 | 4032 | 8.1W | 18.6x |

Assumes 2mW per MZI phase shifter, 0.1mW per photodetector, 10mW laser source.

## Building for Production

To move from simulation to production photonic hardware:

1. **Hardware Driver Layer** — FPGA control interface for DAC/ADC
2. **Real-time Calibration** — Closed-loop feedback for MZI phase stabilization
3. **Chip Integration** — Partner with foundry (e.g., GlobalFoundries 45CLO)
4. **Edge Deployment** — Package for inference on photonic accelerator boards

## Tech Stack

- **Language:** C++20
- **Build:** CMake + Ninja
- **Math:** Eigen3 (linear algebra)
- **Testing:** Google Test
- **Toolchain:** MSYS2 MinGW-w64 g++

## License

MIT

## Citation

If you use this in research, please cite:

```bibtex
@software{photonic_computing_simulation,
  title={Photonic Computing Simulation: ONN Simulator with Physical Effects},
  author={Griffith},
  year={2026},
  url={https://github.com/Griffith-7/photonic-computing-simulation}
}
```
