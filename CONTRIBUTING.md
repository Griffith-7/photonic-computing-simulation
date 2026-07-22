# Contributing to Photonic Computing Simulation

Thank you for your interest in contributing! This guide will help you get started.

## Getting Started

### Prerequisites

- MSYS2 with MinGW-w64 toolchain
- CMake 3.20+
- Ninja build system
- Eigen3
- Python 3.10+ (for generating benchmark charts)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Griffith-7/photonic-computing-simulation.git
cd photonic-computing-simulation

# Build
export PATH="/c/msys64/mingw64/bin:/c/msys64/usr/bin:$PATH"
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
ninja -C build

# Run tests
./build/onn_tests
```

## Project Structure

```
src/
├── core/           # Types and constants
├── photonic/       # MZI mesh architectures
├── physical/       # Physical effects simulation
├── network/        # Neural network components
└── data/           # Dataset handling
tests/
├── test_mzi.cpp        # MZI unit tests
├── test_mesh.cpp       # Mesh decomposition tests
├── test_physical.cpp   # Physical effects tests
├── test_network.cpp    # Network/pipeline tests
└── test_e2e.cpp        # End-to-end tests
benchmarks/
└── generate_charts.py  # Benchmark chart generation
```

## Code Style

- **Naming**: `snake_case` for functions/variables, `CamelCase` for classes
- **Namespace**: `onn::` for all project code
- **Headers**: Use `#pragma once`
- **Includes**: System headers first, then project headers
- **Formatting**: 4 spaces indentation, no tabs

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Write Tests

All new functionality must include unit tests. Add tests to the appropriate `tests/test_*.cpp` file.

### 3. Run Tests

```bash
ninja -C build
./build/onn_tests
```

All 38 tests must pass before submitting a PR.

### 4. Update Documentation

- Update `README.md` if adding new features
- Update `TARGETS.md` if changing performance characteristics
- Add comments for complex algorithms

### 5. Submit PR

- Write a clear PR description
- Reference any related issues
- Ensure CI passes

## Adding New Physical Effects

1. Create `src/physical/your_effect.h` and `src/physical/your_effect.cpp`
2. Add to `PhysicalConfig` struct in `physical_layer.h`
3. Integrate into `LinearLayer::forward()` in `linear_layer.cpp`
4. Add unit tests in `tests/test_physical.cpp`
5. Add to benchmark in `src/main.cpp`

## Adding New Benchmarks

1. Add benchmark function in `src/main.cpp`
2. Add command-line flag in `main()`
3. Update `print_usage()` with new flag
4. Generate charts: `python benchmarks/generate_charts.py`

## Reporting Issues

- Use GitHub Issues
- Include reproduction steps
- Include system info (OS, compiler version, CMake version)
- Attach benchmark output if relevant

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
