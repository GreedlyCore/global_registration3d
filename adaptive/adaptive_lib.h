#pragma once

#include <string>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using CloudPtr = CloudT::Ptr;

struct GeometricBootstrappingParams {
    double kappa_spheric = 0.10;
    double kappa_disc = 0.15;
    double tau_v = 0.05;
    double delta_v = 0.10;

    double tau_l = 0.005;
    double tau_m = 0.02;
    double tau_g = 0.05;
    int N_r = 2000;
    double r_max = 5.0;
};

struct GeometricBootstrappingResult {
    double voxel_size = 0.1;
    double sphericity = 0.0;
    double spread_s = 0.0;
    Eigen::Vector3d eigenvalues = Eigen::Vector3d::Zero();
    Eigen::Matrix3d eigenvectors = Eigen::Matrix3d::Identity();

    double r_local = 0.0;
    double r_middle = 0.0;
    double r_global = 0.0;
};

GeometricBootstrappingResult geometricBootstrappingPipeline(
    const CloudPtr& cloud_src,
    const CloudPtr& cloud_tgt,
    const GeometricBootstrappingParams& params);

CloudPtr loadKittiPointCloud(const std::string& filepath);
CloudPtr loadMulranPointCloud(const std::string& filepath);
CloudPtr loadOxfordPointCloud(const std::string& filepath);
CloudPtr loadPointCloudByDataset(const std::string& filepath, const std::string& dataset);
