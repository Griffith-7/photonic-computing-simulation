"""Smoke tests for onn Python bindings."""

import numpy as np
import pytest

try:
    import onn
except ImportError:
    pytest.skip("onn Python module not built", allow_module_level=True)


def test_version():
    assert hasattr(onn, '__version__')
    assert onn.__version__ == "0.1.0"


def test_constants():
    assert abs(onn.PI - 3.14159265358979) < 1e-10
    assert abs(onn.TWO_PI - 2 * onn.PI) < 1e-10
    assert onn.TELECOM_WAVELENGTH == 1550e-9


def test_mzi_unitarity():
    mzi = onn.MZI(theta=0.5, phi=1.2)
    U = mzi.transfer_matrix()
    assert U.shape == (2, 2)
    product = U.conj().T @ U
    identity = np.eye(2)
    err = np.linalg.norm(product - identity)
    assert err < 1e-12


def test_mzi_bar_cross():
    bar = onn.MZI.bar_state()
    cross = onn.MZI.cross_state()
    assert bar.is_bar_state()
    assert cross.is_cross_state()


def test_mzi_properties():
    mzi = onn.MZI(theta=1.0, phi=2.0)
    assert abs(mzi.theta - 1.0) < 1e-10
    assert abs(mzi.phi - 2.0) < 1e-10
    mzi.theta = 0.5
    assert abs(mzi.theta - 0.5) < 1e-10


def test_reck_decomposition():
    np.random.seed(42)
    N = 4
    raw = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    Q, _ = np.linalg.qr(raw)
    mzis = onn.ReckDecomposer.decompose(Q.astype(np.complex128))
    assert len(mzis) == N * (N - 1) // 2
    reconstructed = onn.ReckDecomposer.reconstruct(mzis, N)
    err = onn.ReckDecomposer.error(Q.astype(np.complex128), mzis)
    assert err < 1e-8


def test_clements_decomposition():
    np.random.seed(42)
    N = 4
    raw = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    Q, _ = np.linalg.qr(raw)
    mzis = onn.ClementsDecomposer.decompose(Q.astype(np.complex128))
    assert len(mzis) == N * (N - 1) // 2
    err = onn.ClementsDecomposer.error(Q.astype(np.complex128), mzis)
    assert err < 1e-8


def test_svd_mapper():
    np.random.seed(42)
    W = np.random.randn(8, 8) + 1j * np.random.randn(8, 8)
    mapping = onn.SVDMapper.map_to_optical(W, onn.MeshType.CLEMENTS)
    assert mapping.N == 8
    assert len(mapping.u_mzis) > 0
    input_vec = np.random.randn(8) + 1j * np.random.randn(8)
    output = onn.SVDMapper.forward(mapping, input_vec.astype(np.complex128))
    expected = W @ input_vec
    err = np.linalg.norm(output - expected) / np.linalg.norm(expected)
    assert err < 1e-8


def test_physical_config():
    cfg = onn.PhysicalConfig()
    assert cfg.drift_enabled is True
    assert abs(cfg.drift_sigma - 0.05) < 1e-10
    cfg.drift_sigma = 0.10
    assert abs(cfg.drift_sigma - 0.10) < 1e-10


def test_train_config():
    cfg = onn.TrainConfig()
    assert cfg.epochs == 10
    assert cfg.batch_size == 32
    cfg.epochs = 5
    assert cfg.epochs == 5


def test_onn_forward():
    configs = [
        onn.LayerConfig(16, 32, onn.ActivationType.RELU, onn.MeshType.CLEMENTS),
        onn.LayerConfig(32, 10, onn.ActivationType.NONE, onn.MeshType.CLEMENTS),
    ]
    net = onn.ONN(configs, False)
    assert net.num_layers() == 2
    input_vec = np.random.randn(16)
    output = net.forward(input_vec)
    assert output.shape == (10,)
    assert abs(output.sum() - 1.0) < 1e-6


def test_onn_ideal_forward():
    configs = [
        onn.LayerConfig(8, 16, onn.ActivationType.RELU, onn.MeshType.CLEMENTS),
        onn.LayerConfig(16, 10, onn.ActivationType.NONE, onn.MeshType.CLEMENTS),
    ]
    net = onn.ONN(configs, False)
    input_vec = np.random.randn(8)
    output_ideal = net.forward_ideal(input_vec)
    output_mesh = net.forward(input_vec)
    assert output_ideal.shape == (10,)
    assert output_mesh.shape == (10,)


def test_mnist_loader_synthetic():
    data = onn.MNISTLoader.generate_synthetic(100, 16, 10, 42)
    assert data.num_samples == 100
    assert data.image_size == 16
    assert data.images.shape == (100, 16)
    assert data.labels.shape == (100,)
    assert data.labels.min() >= 0
    assert data.labels.max() <= 9


def test_mnist_normalize():
    vec = np.random.uniform(0, 255, size=784)
    normed = onn.MNISTLoader.normalize(vec)
    assert normed.max() <= 1.0 + 1e-10
    assert normed.min() >= 0.0 - 1e-10
    np.testing.assert_array_almost_equal(normed, vec / 255.0)


def test_activation_relu():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = onn.Activation.relu(x)
    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_activation_softmax():
    x = np.array([1.0, 2.0, 3.0])
    result = onn.Activation.softmax(x)
    assert abs(result.sum() - 1.0) < 1e-10
    assert result[2] > result[0]


def test_drift_compensator():
    cfg = onn.CalibrationConfig()
    cfg.max_iterations = 5
    comp = onn.DriftCompensator(cfg)
    assert comp.iterations_used() == 0
    assert comp.final_error() == 0.0


def test_physical_layer():
    cfg = onn.PhysicalConfig()
    cfg.drift_enabled = True
    cfg.drift_sigma = 0.02
    phys = onn.PhysicalLayer(cfg)
    assert phys.config().drift_enabled is True


def test_layer_config_defaults():
    cfg = onn.LayerConfig(64, 32)
    assert cfg.in_features == 64
    assert cfg.out_features == 32
    assert cfg.activation == onn.ActivationType.RELU
    assert cfg.mesh == onn.MeshType.CLEMENTS
