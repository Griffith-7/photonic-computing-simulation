#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "core/types.h"
#include "photonic/mzi.h"
#include "photonic/reck.h"
#include "photonic/clements.h"
#include "photonic/svd_mapper.h"
#include "physical/physical_layer.h"
#include "physical/drift_compensation.h"
#include "network/network.h"
#include "network/linear_layer.h"
#include "network/trainer.h"
#include "network/activation.h"
#include "data/mnist_loader.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(onn_python, m) {
    m.doc() = "Optical Neural Network Simulator - Python Bindings";
    m.attr("__version__") = "0.1.0";

    // --- Constants ---
    m.attr("PI") = onn::PI;
    m.attr("TWO_PI") = onn::TWO_PI;
    m.attr("TELECOM_WAVELENGTH") = onn::TELECOM_WAVELENGTH;
    m.attr("SPEED_OF_LIGHT") = onn::SPEED_OF_LIGHT;

    // --- Enums ---
    py::enum_<onn::ActivationType>(m, "ActivationType")
        .value("RELU", onn::ActivationType::RELU)
        .value("SIGMOID", onn::ActivationType::SIGMOID)
        .value("TANH", onn::ActivationType::TANH)
        .value("SOFTMAX", onn::ActivationType::SOFTMAX)
        .value("NONE", onn::ActivationType::NONE);

    py::enum_<onn::MeshType>(m, "MeshType")
        .value("RECK", onn::MeshType::RECK)
        .value("CLEMENTS", onn::MeshType::CLEMENTS);

    // --- POD structs ---
    py::class_<onn::MZIRecord>(m, "MZIRecord")
        .def(py::init<>())
        .def_readwrite("mode1", &onn::MZIRecord::mode1)
        .def_readwrite("mode2", &onn::MZIRecord::mode2)
        .def_readwrite("theta", &onn::MZIRecord::theta)
        .def_readwrite("phi", &onn::MZIRecord::phi);

    py::class_<onn::MZIPosition>(m, "MZIPosition")
        .def(py::init<>())
        .def_readwrite("row", &onn::MZIPosition::row)
        .def_readwrite("col", &onn::MZIPosition::col);

    py::class_<onn::Wave>(m, "Wave")
        .def(py::init<>())
        .def(py::init<onn::Complex, double>(),
             py::arg("amplitude") = onn::Complex{0.0, 0.0},
             py::arg("wavelength") = onn::TELECOM_WAVELENGTH)
        .def_readwrite("amplitude", &onn::Wave::amplitude)
        .def_readwrite("wavelength", &onn::Wave::wavelength)
        .def("intensity", &onn::Wave::intensity)
        .def("phase", &onn::Wave::phase)
        .def("power_dbm", &onn::Wave::power_dbm)
        .def("normalized", &onn::Wave::normalized);

    // --- Config structs ---
    py::class_<onn::PhysicalConfig>(m, "PhysicalConfig")
        .def(py::init<>())
        .def_readwrite("drift_enabled", &onn::PhysicalConfig::drift_enabled)
        .def_readwrite("drift_sigma", &onn::PhysicalConfig::drift_sigma)
        .def_readwrite("attenuation_enabled", &onn::PhysicalConfig::attenuation_enabled)
        .def_readwrite("attenuation_db_per_cm", &onn::PhysicalConfig::attenuation_db_per_cm)
        .def_readwrite("coupling_loss_db", &onn::PhysicalConfig::coupling_loss_db)
        .def_readwrite("shot_noise_enabled", &onn::PhysicalConfig::shot_noise_enabled)
        .def_readwrite("quantum_efficiency", &onn::PhysicalConfig::quantum_efficiency)
        .def_readwrite("crosstalk_enabled", &onn::PhysicalConfig::crosstalk_enabled)
        .def_readwrite("crosstalk_kappa", &onn::PhysicalConfig::crosstalk_kappa)
        .def_readwrite("crosstalk_sigma", &onn::PhysicalConfig::crosstalk_sigma)
        .def_readwrite("time_step", &onn::PhysicalConfig::time_step)
        .def_readwrite("seed", &onn::PhysicalConfig::seed);

    py::class_<onn::CalibrationConfig>(m, "CalibrationConfig")
        .def(py::init<>())
        .def_readwrite("learning_rate", &onn::CalibrationConfig::learning_rate)
        .def_readwrite("perturbation_delta", &onn::CalibrationConfig::perturbation_delta)
        .def_readwrite("max_iterations", &onn::CalibrationConfig::max_iterations)
        .def_readwrite("convergence_threshold", &onn::CalibrationConfig::convergence_threshold);

    py::class_<onn::TrainConfig>(m, "TrainConfig")
        .def(py::init<>())
        .def_readwrite("epochs", &onn::TrainConfig::epochs)
        .def_readwrite("batch_size", &onn::TrainConfig::batch_size)
        .def_readwrite("learning_rate", &onn::TrainConfig::learning_rate)
        .def_readwrite("hidden_layers", &onn::TrainConfig::hidden_layers)
        .def_readwrite("mesh", &onn::TrainConfig::mesh);

    py::class_<onn::LayerConfig>(m, "LayerConfig")
        .def(py::init<int, int, onn::ActivationType, onn::MeshType>(),
             py::arg("in_features"), py::arg("out_features"),
             py::arg("activation") = onn::ActivationType::RELU,
             py::arg("mesh") = onn::MeshType::CLEMENTS)
        .def_readwrite("in_features", &onn::LayerConfig::in_features)
        .def_readwrite("out_features", &onn::LayerConfig::out_features)
        .def_readwrite("activation", &onn::LayerConfig::activation)
        .def_readwrite("mesh", &onn::LayerConfig::mesh);

    // --- Result structs ---
    py::class_<onn::SVDResult>(m, "SVDResult")
        .def_readonly("U", &onn::SVDResult::U)
        .def_readonly("S", &onn::SVDResult::S)
        .def_readonly("Vh", &onn::SVDResult::Vh);

    py::class_<onn::OpticalMapping>(m, "OpticalMapping")
        .def_readonly("u_mzis", &onn::OpticalMapping::u_mzis)
        .def_readonly("singular_values", &onn::OpticalMapping::singular_values)
        .def_readonly("vh_mzis", &onn::OpticalMapping::vh_mzis)
        .def_readonly("N", &onn::OpticalMapping::N)
        .def_readonly("N_out", &onn::OpticalMapping::N_out)
        .def_readonly("mesh", &onn::OpticalMapping::mesh);

    py::class_<onn::TrainResult>(m, "TrainResult")
        .def(py::init<>())
        .def_readonly("train_loss", &onn::TrainResult::train_loss)
        .def_readonly("test_accuracy", &onn::TrainResult::test_accuracy)
        .def_readonly("input_size", &onn::TrainResult::input_size)
        .def_readonly("num_classes", &onn::TrainResult::num_classes);

    py::class_<onn::MNISTData>(m, "MNISTData")
        .def_readonly("images", &onn::MNISTData::images)
        .def_readonly("labels", &onn::MNISTData::labels)
        .def_readonly("num_samples", &onn::MNISTData::num_samples)
        .def_readonly("image_size", &onn::MNISTData::image_size);

    // --- MZI ---
    py::class_<onn::MZI>(m, "MZI")
        .def(py::init<double, double>(),
             py::arg("theta") = 0.0, py::arg("phi") = 0.0)
        .def_property("theta", &onn::MZI::theta, &onn::MZI::set_theta)
        .def_property("phi", &onn::MZI::phi, &onn::MZI::set_phi)
        .def("transfer_matrix", &onn::MZI::transfer_matrix)
        .def("is_bar_state", &onn::MZI::is_bar_state, py::arg("tol") = 1e-10)
        .def("is_cross_state", &onn::MZI::is_cross_state, py::arg("tol") = 1e-10)
        .def_static("bar_state", &onn::MZI::bar_state)
        .def_static("cross_state", &onn::MZI::cross_state);

    // --- Decomposers ---
    py::class_<onn::ReckDecomposer>(m, "ReckDecomposer")
        .def_static("decompose", &onn::ReckDecomposer::decompose)
        .def_static("reconstruct", &onn::ReckDecomposer::reconstruct)
        .def_static("error", &onn::ReckDecomposer::error);

    py::class_<onn::ClementsDecomposer>(m, "ClementsDecomposer")
        .def_static("decompose", &onn::ClementsDecomposer::decompose)
        .def_static("reconstruct", &onn::ClementsDecomposer::reconstruct)
        .def_static("error", &onn::ClementsDecomposer::error);

    // --- SVD Mapper ---
    py::class_<onn::SVDMapper>(m, "SVDMapper")
        .def_static("decompose", &onn::SVDMapper::decompose)
        .def_static("map_to_optical", &onn::SVDMapper::map_to_optical,
                     py::arg("W"), py::arg("mesh") = onn::MeshType::CLEMENTS)
        .def_static("forward", &onn::SVDMapper::forward)
        .def_static("forward_with_loss", &onn::SVDMapper::forward_with_loss);

    // --- Physical layer ---
    py::class_<onn::PhysicalLayer>(m, "PhysicalLayer")
        .def(py::init<const onn::PhysicalConfig&>(),
             py::arg("config") = onn::PhysicalConfig{})
        .def("apply", &onn::PhysicalLayer::apply)
        .def("advance_time", &onn::PhysicalLayer::advance_time)
        .def("reset", &onn::PhysicalLayer::reset)
        .def("config", &onn::PhysicalLayer::config, py::return_value_policy::reference_internal);

    // --- Drift Compensation ---
    py::class_<onn::DriftCompensator>(m, "DriftCompensator")
        .def(py::init<const onn::CalibrationConfig&>(),
             py::arg("config") = onn::CalibrationConfig{})
        .def("calibrate", &onn::DriftCompensator::calibrate)
        .def("iterations_used", &onn::DriftCompensator::iterations_used)
        .def("final_error", &onn::DriftCompensator::final_error);

    // --- LinearLayer ---
    py::class_<onn::LinearLayer>(m, "LinearLayer")
        .def(py::init<int, int, onn::MeshType, bool, onn::PhysicalConfig>(),
             py::arg("in_features"), py::arg("out_features"),
             py::arg("mesh") = onn::MeshType::CLEMENTS,
             py::arg("physical") = false,
             py::arg("phys_config") = onn::PhysicalConfig{})
        .def("forward", &onn::LinearLayer::forward)
        .def("forward_ideal", &onn::LinearLayer::forward_ideal)
        .def("weight_matrix", &onn::LinearLayer::weight_matrix,
             py::return_value_policy::reference_internal)
        .def("set_weight_matrix", &onn::LinearLayer::set_weight_matrix)
        .def_property_readonly("in_features", &onn::LinearLayer::in_features)
        .def_property_readonly("out_features", &onn::LinearLayer::out_features);

    // --- ONN Network ---
    py::class_<onn::ONN>(m, "ONN")
        .def(py::init<const std::vector<onn::LayerConfig>&, bool, onn::PhysicalConfig>(),
             py::arg("layers"),
             py::arg("physical") = false,
             py::arg("phys_config") = onn::PhysicalConfig{})
        .def("forward", &onn::ONN::forward)
        .def("forward_ideal", &onn::ONN::forward_ideal)
        .def("load_weights", &onn::ONN::load_weights)
        .def("compensate", &onn::ONN::compensate,
             py::arg("X_ref"), py::arg("y_ref"),
             py::arg("cal_config") = onn::CalibrationConfig{})
        .def("num_layers", &onn::ONN::num_layers)
        .def("layer", &onn::ONN::layer, py::return_value_policy::reference_internal);

    // --- Trainer ---
    py::class_<onn::Trainer>(m, "Trainer")
        .def_static("train_digitall_twin", &onn::Trainer::train_digitall_twin)
        .def_static("create_optical_network",
            py::overload_cast<const onn::TrainResult&, bool, const onn::PhysicalConfig&>(
                &onn::Trainer::create_optical_network),
            py::arg("result"),
            py::arg("physical") = false,
            py::arg("phys_config") = onn::PhysicalConfig{})
        .def_static("predict", &onn::Trainer::predict)
        .def_static("predict_ideal", &onn::Trainer::predict_ideal)
        .def_static("accuracy", &onn::Trainer::accuracy)
        .def_static("ideal_accuracy", &onn::Trainer::ideal_accuracy);

    // --- Activation ---
    py::class_<onn::Activation>(m, "Activation")
        .def_static("relu", &onn::Activation::relu)
        .def_static("sigmoid", &onn::Activation::sigmoid)
        .def_static("softmax", &onn::Activation::softmax);

    // --- MNIST Loader ---
    py::class_<onn::MNISTLoader>(m, "MNISTLoader")
        .def_static("load", &onn::MNISTLoader::load)
        .def_static("load_subset", &onn::MNISTLoader::load_subset)
        .def_static("generate_synthetic", &onn::MNISTLoader::generate_synthetic,
                     py::arg("num_samples") = 200,
                     py::arg("image_size") = 16,
                     py::arg("num_classes") = 10,
                     py::arg("seed") = 42)
        .def_static("normalize", &onn::MNISTLoader::normalize)
        .def_static("normalize_batch", &onn::MNISTLoader::normalize_batch);
}
