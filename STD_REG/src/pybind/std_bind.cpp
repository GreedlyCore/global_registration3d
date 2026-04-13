#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../../include/STDesc.h"

namespace py = pybind11;

namespace {

pcl::PointCloud<pcl::PointXYZI>::Ptr points_from_python(const py::sequence &seq) {
  auto cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  cloud->reserve(seq.size());
  for (const auto &item : seq) {
    py::sequence point = py::reinterpret_borrow<py::sequence>(item);
    if (point.size() < 3) {
      throw std::runtime_error("Each point must contain at least x, y, z");
    }
    pcl::PointXYZI pcl_point;
    pcl_point.x = point[0].cast<float>();
    pcl_point.y = point[1].cast<float>();
    pcl_point.z = point[2].cast<float>();
    pcl_point.intensity = point.size() > 3 ? point[3].cast<float>() : 0.0f;
    cloud->push_back(pcl_point);
  }
  return cloud;
}

py::dict correspondence_to_dict(const std::pair<STDesc, STDesc> &pair) {
  py::dict result;
  result["source_triangle"] = py::make_tuple(
      py::make_tuple(pair.first.vertex_A_.x(), pair.first.vertex_A_.y(), pair.first.vertex_A_.z()),
      py::make_tuple(pair.first.vertex_B_.x(), pair.first.vertex_B_.y(), pair.first.vertex_B_.z()),
      py::make_tuple(pair.first.vertex_C_.x(), pair.first.vertex_C_.y(), pair.first.vertex_C_.z()));
  result["target_triangle"] = py::make_tuple(
      py::make_tuple(pair.second.vertex_A_.x(), pair.second.vertex_A_.y(), pair.second.vertex_A_.z()),
      py::make_tuple(pair.second.vertex_B_.x(), pair.second.vertex_B_.y(), pair.second.vertex_B_.z()),
      py::make_tuple(pair.second.vertex_C_.x(), pair.second.vertex_C_.y(), pair.second.vertex_C_.z()));
  result["source_center"] = py::make_tuple(pair.first.center_.x(), pair.first.center_.y(), pair.first.center_.z());
  result["target_center"] = py::make_tuple(pair.second.center_.x(), pair.second.center_.y(), pair.second.center_.z());
  return result;
}

py::list expanded_correspondences(const std::vector<std::pair<STDesc, STDesc>> &pairs) {
  py::list output;
  for (const auto &pair : pairs) {
    output.append(py::make_tuple(
        py::make_tuple(pair.first.vertex_A_.x(), pair.first.vertex_A_.y(), pair.first.vertex_A_.z()),
        py::make_tuple(pair.second.vertex_A_.x(), pair.second.vertex_A_.y(), pair.second.vertex_A_.z())));
    output.append(py::make_tuple(
        py::make_tuple(pair.first.vertex_B_.x(), pair.first.vertex_B_.y(), pair.first.vertex_B_.z()),
        py::make_tuple(pair.second.vertex_B_.x(), pair.second.vertex_B_.y(), pair.second.vertex_B_.z())));
    output.append(py::make_tuple(
        py::make_tuple(pair.first.vertex_C_.x(), pair.first.vertex_C_.y(), pair.first.vertex_C_.z()),
        py::make_tuple(pair.second.vertex_C_.x(), pair.second.vertex_C_.y(), pair.second.vertex_C_.z())));
  }
  return output;
}

}

PYBIND11_MODULE(std_solver, m) {
  m.doc() = "Standalone STD registration bindings";

  py::class_<ConfigSetting>(m, "ConfigSetting")
      .def(py::init<>())
      .def_readwrite("stop_skip_enable_", &ConfigSetting::stop_skip_enable_)
      .def_readwrite("ds_size_", &ConfigSetting::ds_size_)
      .def_readwrite("maximum_corner_num_", &ConfigSetting::maximum_corner_num_)
      .def_readwrite("plane_merge_normal_thre_", &ConfigSetting::plane_merge_normal_thre_)
      .def_readwrite("plane_merge_dis_thre_", &ConfigSetting::plane_merge_dis_thre_)
      .def_readwrite("plane_detection_thre_", &ConfigSetting::plane_detection_thre_)
      .def_readwrite("voxel_size_", &ConfigSetting::voxel_size_)
      .def_readwrite("voxel_init_num_", &ConfigSetting::voxel_init_num_)
      .def_readwrite("proj_image_resolution_", &ConfigSetting::proj_image_resolution_)
      .def_readwrite("proj_dis_min_", &ConfigSetting::proj_dis_min_)
      .def_readwrite("proj_dis_max_", &ConfigSetting::proj_dis_max_)
      .def_readwrite("corner_thre_", &ConfigSetting::corner_thre_)
      .def_readwrite("descriptor_near_num_", &ConfigSetting::descriptor_near_num_)
      .def_readwrite("descriptor_min_len_", &ConfigSetting::descriptor_min_len_)
      .def_readwrite("descriptor_max_len_", &ConfigSetting::descriptor_max_len_)
      .def_readwrite("non_max_suppression_radius_", &ConfigSetting::non_max_suppression_radius_)
      .def_readwrite("std_side_resolution_", &ConfigSetting::std_side_resolution_)
      .def_readwrite("skip_near_num_", &ConfigSetting::skip_near_num_)
      .def_readwrite("candidate_num_", &ConfigSetting::candidate_num_)
      .def_readwrite("sub_frame_num_", &ConfigSetting::sub_frame_num_)
      .def_readwrite("rough_dis_threshold_", &ConfigSetting::rough_dis_threshold_)
      .def_readwrite("vertex_diff_threshold_", &ConfigSetting::vertex_diff_threshold_)
      .def_readwrite("icp_threshold_", &ConfigSetting::icp_threshold_)
      .def_readwrite("normal_threshold_", &ConfigSetting::normal_threshold_)
      .def_readwrite("dis_threshold_", &ConfigSetting::dis_threshold_);

  py::class_<STDesc>(m, "STDesc")
      .def(py::init<>())
      .def_readwrite("side_length_", &STDesc::side_length_)
      .def_readwrite("angle_", &STDesc::angle_)
      .def_readwrite("center_", &STDesc::center_)
      .def_readwrite("frame_id_", &STDesc::frame_id_)
      .def_readwrite("vertex_A_", &STDesc::vertex_A_)
      .def_readwrite("vertex_B_", &STDesc::vertex_B_)
      .def_readwrite("vertex_C_", &STDesc::vertex_C_)
      .def_readwrite("vertex_attached_", &STDesc::vertex_attached_);

  py::class_<STDescManager>(m, "STDescManager")
      .def(py::init<>())
      .def(py::init([](const ConfigSetting &config) {
        return STDescManager(const_cast<ConfigSetting &>(config));
      }))
      .def("set_config", [](STDescManager &self, const ConfigSetting &config) { self.config_setting_ = config; })
      .def("match_pairwise",
           [](STDescManager &self, const py::sequence &source_points,
              const py::sequence &target_points) {
             auto source_cloud = points_from_python(source_points);
             auto target_cloud = points_from_python(target_points);
           std::pair<Eigen::Vector3d, Eigen::Matrix3d> transform;
           transform.first.setZero();
           transform.second.setIdentity();
             std::vector<std::pair<STDesc, STDesc>> match_pairs;
             double score = 0.0;
             double downsample_ms = 0.0;
             double feature_ms = 0.0;
             double correspondence_ms = 0.0;
             double registration_ms = 0.0;
             bool ok = self.MatchPairwise(source_cloud, target_cloud, transform,
                                          match_pairs, score, downsample_ms,
                                          feature_ms, correspondence_ms,
                                          registration_ms);

             py::dict result;
             result["ok"] = ok;
             result["score"] = score;
             if (!ok) {
               transform.first.setZero();
               transform.second.setIdentity();
             }
             result["transform_translation"] = py::make_tuple(transform.first.x(), transform.first.y(), transform.first.z());
             result["transform_rotation"] = transform.second;
             py::list triangle_pairs;
             for (const auto &pair : match_pairs) {
               triangle_pairs.append(correspondence_to_dict(pair));
             }
             result["triangle_pairs"] = triangle_pairs;
             result["correspondences"] = expanded_correspondences(match_pairs);
             py::dict timings;
             timings["downsample"] = downsample_ms;
             timings["feature"] = feature_ms;
             timings["candidate"] = correspondence_ms;
             timings["registration"] = registration_ms;
             result["timings_ms"] = timings;
             return result;
           },
           py::arg("source_points"), py::arg("target_points"));
}
