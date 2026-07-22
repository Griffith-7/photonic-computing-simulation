# Physics Model

## Overview

The simulator models a photonic neural network based on Mach-Zehnder Interferometer (MZI) meshes. Each MZI implements a 2×2 unitary transformation, and cascaded MZIs form larger unitary matrices.

## MZI Transfer Matrix

A single MZI with internal phase θ and external phase φ:

```
U_MZI(θ, φ) = [ cos(θ/2)    -sin(θ/2) ] × [ 1    0  ]
               [ sin(θ/2)     cos(θ/2) ]   [ 0  e^iφ ]
```

Properties:
- **Unitary**: U†U = I (power conservation)
- **Determinant**: det(U) = e^iφ
- **Parameters**: 2 per MZI (θ, φ)

## Mesh Topologies

### Reck Decomposition
- Triangular mesh structure
- N(N-1)/2 MZIs for N×N unitary
- Lower bounded: each MZI affects a triangular region
- Better for cascaded architectures

### Clements Decomposition
- Rectangular mesh structure
- N(N-1)/2 MZIs for N×N unitary
- More balanced path lengths
- Better for fabrication (uniform waveguide lengths)

## SVD Optical Mapping

Any weight matrix W ∈ ℝ^{m×n} can be mapped to photonic hardware:

```
W = U Σ V†
```

Where:
- U ∈ ℂ^{m×m} unitary → implemented as Reck/Clements mesh
- Σ ∈ ℝ^{m×n} diagonal → implemented as variable attenuators
- V† ∈ ℂ^{n×n} unitary → implemented as Reck/Clements mesh

## Physical Effects

### 1. Thermal Drift

Thermally-induced phase noise in MZI waveguides:

```
W_eff = W + D ⊙ W
```

Where D(i,j) ~ N(0, σ²) is the drift matrix.

**Key properties:**
- Persistent per layer (not re-sampled per inference)
- Models Brownian motion of thermo-optic phase shifters
- σ = 0.10 represents 10% weight perturbation (physically realistic for unstabilized MZIs)

**Physical origin:**
- Silicon thermo-optic coefficient: dn/dT ≈ 1.86 × 10⁻⁴ K⁻¹
- Waveguide phase: φ = (2π/λ) × n_eff × L
- Temperature fluctuation ΔT causes phase drift Δφ ∝ ΔT

### 2. Waveguide Attenuation

Loss from coupling and propagation:

```
total_loss_dB = coupling_loss × 2 + α × L
attenuation = 10^(-total_loss_dB / 20) × (1 + ε)
```

Where ε ~ N(0, 0.03²) is per-element variation.

**Parameters:**
| Parameter | Value | Source |
|-----------|-------|--------|
| Coupling loss | 0.1 dB/facet | Fiber-to-chip coupling |
| Propagation loss | 0.3 dB/cm | Silicon waveguide |
| Per-element variation | ±3% | Manufacturing tolerance |

### 3. Shot Noise

Quantum noise from photon detection:

```
P_detected ~ Poisson(η × P_signal × N_photons)
```

Where η is quantum efficiency.

**Model:**
- Each output mode generates photoelectrons
- Electron count follows Poisson distribution
- Power reconstructed from electron count
- Dominant at low power levels

### 4. Thermal Crosstalk

Heat diffusion between adjacent waveguides:

```
output(i) += κ × Σ_j exp(-|i-j|²/2σ²) × output(j)
```

**Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| κ | 0.05 | Coupling strength |
| σ | 2.0 modes | Spatial decay length |

**Physical origin:**
- Thermo-optic effect heats nearby waveguides
- Coupling decreases exponentially with distance
- σ depends on waveguide spacing and substrate thermal conductivity

## Drift Compensation

### Per-Element Least-Squares

1. Collect calibration data: inputs X, ideal outputs Y_ideal, physical outputs Y_phys
2. Compute output perturbation: ΔY = Y_phys - Y_ideal
3. Solve for weight correction: ΔW = ΔY × (X^H X)^{-1} X^H
4. Apply correction: W_corrected(i,j) = W(i,j) / (1 + ΔW(i,j)/W(i,j))

**Complexity:** O(K³ + K²×fan_in) per layer, where K = calibration samples

**Effectiveness:**
- Small network (128): 98.5% → 99.0% (+0.5%)
- Medium network (256,128): 99.9% → 99.3% (full recovery)

## Power Model

```
P_total = N_MZIs × P_MZI + N_detectors × P_detector + P_laser
```

| Component | Power |
|-----------|-------|
| MZI phase shifter | 2 mW |
| Photodetector | 0.1 mW |
| Laser source | 10 mW |

## References

1. Clements, W.R. et al. "Optimal design for universal multiport interferometers." *Optica* 3.12 (2016): 1460-1465.
2. Reck, M. et al. "Experimental realization of any discrete variable unitary operator." *Physical Review Letters* 73.1 (1994): 58.
3. Shen, Y. et al. "Deep learning with coherent nanophotonic circuits." *Nature Photonics* 11.7 (2017): 441-446.
4. Feldmann, J. et al. "All-optical non-volatile multi-level switching in phase-change materials." *Nature Communications* 12.1 (2021): 1-8.
