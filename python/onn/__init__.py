"""
Photonic Computing Simulation - Python Bindings

A high-performance C++20 simulator for optical neural networks
based on Mach-Zehnder Interferometer (MZI) mesh architectures.
"""

from .onn_python import (
    # Version
    __version__,

    # Constants
    PI, TWO_PI, TELECOM_WAVELENGTH, SPEED_OF_LIGHT,

    # Enums
    ActivationType, MeshType,

    # POD structs
    MZIRecord, MZIPosition, Wave,

    # Config structs
    PhysicalConfig, CalibrationConfig, TrainConfig, LayerConfig,

    # Result structs
    SVDResult, OpticalMapping, TrainResult, MNISTData,

    # Core classes
    MZI, ReckDecomposer, ClementsDecomposer, SVDMapper,

    # Physical
    PhysicalLayer, DriftCompensator,

    # Network
    LinearLayer, ONN, Trainer, Activation,

    # Data
    MNISTLoader,
)

__all__ = [
    '__version__',
    'PI', 'TWO_PI', 'TELECOM_WAVELENGTH', 'SPEED_OF_LIGHT',
    'ActivationType', 'MeshType',
    'MZIRecord', 'MZIPosition', 'Wave',
    'PhysicalConfig', 'CalibrationConfig', 'TrainConfig', 'LayerConfig',
    'SVDResult', 'OpticalMapping', 'TrainResult', 'MNISTData',
    'MZI', 'ReckDecomposer', 'ClementsDecomposer', 'SVDMapper',
    'PhysicalLayer', 'DriftCompensator',
    'LinearLayer', 'ONN', 'Trainer', 'Activation',
    'MNISTLoader',
]
