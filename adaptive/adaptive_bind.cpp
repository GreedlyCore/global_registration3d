#include <stdexcept>
#include <string>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "adaptive_lib.h"

namespace py = pybind11;

namespace {

py::dict dictFromResult(const GeometricBootstrappingResult& result) {
    py::dict d;
    d["voxel_size"] = result.voxel_size;
    d["r_local"] = result.r_local;
    d["r_middle"] = result.r_middle;
    d["r_global"] = result.r_global;
    d["sphericity"] = result.sphericity;
    d["spread_s"] = result.spread_s;
    d["eigenvalues"] = result.eigenvalues;
    d["eigenvectors"] = result.eigenvectors;
    return d;
}

py::dict getAdaptiveParams(
    const std::string& scan_path,
    const std::string& dataset,
    const double kappa_spheric,
    const double kappa_disc,
    const double tau_v,
    const double delta_v,
    const double tau_l,
    const double tau_m,
    const double tau_g,
    const int N_r,
    const double r_max) {
    CloudPtr cloud = loadPointCloudByDataset(scan_path, dataset);
    if (!cloud || cloud->empty()) {
        throw std::runtime_error("Failed to load point cloud from: " + scan_path);
    }

    GeometricBootstrappingParams params;
    params.kappa_spheric = kappa_spheric;
    params.kappa_disc = kappa_disc;
    params.tau_v = tau_v;
    params.delta_v = delta_v;
    params.tau_l = tau_l;
    params.tau_m = tau_m;
    params.tau_g = tau_g;
    params.N_r = N_r;
    params.r_max = r_max;

    GeometricBootstrappingResult result = geometricBootstrappingPipeline(cloud, cloud, params);
    return dictFromResult(result);
}

}  // namespace

PYBIND11_MODULE(adaptive_bootstrap, m) {
    m.doc() = "Adaptive geometric bootstrapping bindings";

    m.def(
        "get_adaptive_params",
        &getAdaptiveParams,
        py::arg("scan_path"),
        py::arg("dataset") = "kitti",
        py::arg("kappa_spheric") = 0.10,
        py::arg("kappa_disc") = 0.15,
        py::arg("tau_v") = 0.05,
        py::arg("delta_v") = 0.10,
        py::arg("tau_l") = 0.005,
        py::arg("tau_m") = 0.02,
        py::arg("tau_g") = 0.05,
        py::arg("N_r") = 2000,
        py::arg("r_max") = 5.0,
        "Compute adaptive parameters from a single scan file.");
}
