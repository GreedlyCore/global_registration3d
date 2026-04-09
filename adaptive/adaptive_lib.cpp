#include "adaptive_lib.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <omp.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>

namespace {

CloudPtr selectLargerCloud(const CloudPtr& cloud1, const CloudPtr& cloud2) {
    if (cloud1->size() >= cloud2->size()) {
        return cloud1;
    }
    return cloud2;
}

CloudPtr samplePoints(const CloudPtr& cloud, double percentage) {
    if (percentage >= 1.0) {
        return cloud;
    }

    int sample_size = static_cast<int>(cloud->size() * percentage);
    if (sample_size <= 0) {
        sample_size = 1;
    }

    pcl::RandomSample<PointT> random_sample;
    random_sample.setInputCloud(cloud);
    random_sample.setSample(sample_size);

    CloudPtr sampled(new CloudT);
    random_sample.filter(*sampled);

    return sampled;
}

bool computeSphericityMetrics(
    const CloudPtr& cloud,
    double& sphericity,
    double& spread_s,
    Eigen::Vector3d& eigenvalues,
    Eigen::Matrix3d& eigenvectors) {
    if (cloud->empty()) {
        return false;
    }

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    Eigen::Matrix3f cov;
    pcl::computeCovarianceMatrix(*cloud, centroid, cov);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    eigenvalues = solver.eigenvalues().cast<double>();
    eigenvectors = solver.eigenvectors().cast<double>();

    eigenvalues.reverseInPlace();

    Eigen::Matrix3d eigvecs_asc = eigenvectors;
    eigenvectors.col(0) = eigvecs_asc.col(2);
    eigenvectors.col(1) = eigvecs_asc.col(1);
    eigenvectors.col(2) = eigvecs_asc.col(0);

    sphericity = eigenvalues(2) / eigenvalues(0);

    Eigen::Vector3d v3 = eigenvectors.col(2);

    double min_proj = std::numeric_limits<double>::max();
    double max_proj = std::numeric_limits<double>::lowest();

    #pragma omp parallel for reduction(min : min_proj) reduction(max : max_proj)
    for (int i = 0; i < static_cast<int>(cloud->size()); ++i) {
        const PointT& p = (*cloud)[i];
        Eigen::Vector3d point(p.x, p.y, p.z);
        const double proj = point.dot(v3);

        if (proj < min_proj) {
            min_proj = proj;
        }
        if (proj > max_proj) {
            max_proj = proj;
        }
    }

    spread_s = max_proj - min_proj;

    return true;
}

double computeAdaptiveVoxelSize(
    const double sphericity,
    const double spread_s,
    const GeometricBootstrappingParams& params) {
    if (sphericity >= params.tau_v) {
        return params.kappa_spheric * std::sqrt(spread_s);
    }
    return params.kappa_disc * std::sqrt(spread_s);
}

double estimateRadiusForDensity(
    const CloudPtr& cloud,
    const double target_density,
    const double r_max,
    const int num_queries) {
    if (cloud->empty()) {
        return r_max;
    }

    CloudPtr query_points;
    if (cloud->size() > static_cast<size_t>(num_queries)) {
        pcl::RandomSample<PointT> sampler;
        sampler.setInputCloud(cloud);
        sampler.setSample(num_queries);
        query_points.reset(new CloudT);
        sampler.filter(*query_points);
    } else {
        query_points = cloud;
    }

    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);

    double r_low = 0.01;
    double r_high = r_max;
    double r_opt = r_high;
    double best_error = std::numeric_limits<double>::max();

    const int num_iterations = 20;
    const double tolerance = 0.001;

    for (int iter = 0; iter < num_iterations; ++iter) {
        const double r_mid = (r_low + r_high) * 0.5;
        double avg_neighbors = 0.0;

        #pragma omp parallel for reduction(+ : avg_neighbors)
        for (int i = 0; i < static_cast<int>(query_points->size()); ++i) {
            const PointT query = (*query_points)[i];
            std::vector<int> indices;
            std::vector<float> distances;

            if (kdtree.radiusSearch(query, r_mid, indices, distances) > 0) {
                avg_neighbors += static_cast<double>(indices.size());
            }
        }

        avg_neighbors /= static_cast<double>(query_points->size());
        const double expected = target_density * static_cast<double>(cloud->size());
        const double error = std::abs(avg_neighbors - expected);

        if (error < best_error) {
            best_error = error;
            r_opt = r_mid;
        }

        if (avg_neighbors < expected) {
            r_low = r_mid;
        } else {
            r_high = r_mid;
        }

        if (r_high - r_low < tolerance) {
            break;
        }
    }

    return std::min(r_opt, r_max);
}

CloudPtr loadBinPointCloudXYZI(const std::string& filepath, const std::string& dataset_name) {
    CloudPtr cloud(new CloudT);

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << dataset_name << " .bin file: " << filepath << "\n";
        return cloud;
    }

    float fx;
    float fy;
    float fz;
    float reflectance;
    while (file.read(reinterpret_cast<char*>(&fx), sizeof(float))) {
        file.read(reinterpret_cast<char*>(&fy), sizeof(float));
        file.read(reinterpret_cast<char*>(&fz), sizeof(float));
        file.read(reinterpret_cast<char*>(&reflectance), sizeof(float));

        if (file.eof()) {
            break;
        }

        PointT p;
        p.x = static_cast<double>(fx);
        p.y = static_cast<double>(fy);
        p.z = static_cast<double>(fz);
        cloud->push_back(p);
    }

    file.close();
    std::cout << "Loaded " << cloud->size() << " points from " << filepath << "\n";
    return cloud;
}

}  // namespace

GeometricBootstrappingResult geometricBootstrappingPipeline(
    const CloudPtr& cloud_src,
    const CloudPtr& cloud_tgt,
    const GeometricBootstrappingParams& params) {
    GeometricBootstrappingResult result;

    if (!cloud_src || !cloud_tgt || cloud_src->empty() || cloud_tgt->empty()) {
        std::cerr << "Input clouds are empty. Returning default parameters.\n";
        return result;
    }

    // std::cout << "=== Geometric Bootstrapping Pipeline ===\n";

    CloudPtr larger_cloud = selectLargerCloud(cloud_src, cloud_tgt);
    // std::cout << "Selected larger cloud with " << larger_cloud->size() << " points\n";

    CloudPtr sampled_cloud = samplePoints(larger_cloud, params.delta_v);
    // std::cout << "Sampled " << sampled_cloud->size() << " points for sphericity analysis\n";

    const bool pca_success = computeSphericityMetrics(
        sampled_cloud,
        result.sphericity,
        result.spread_s,
        result.eigenvalues,
        result.eigenvectors);

    if (!pca_success) {
        std::cerr << "PCA computation failed!\n";
        result.voxel_size = 0.1;
    } else {
        // std::cout << "Sphericity (lambda3/lambda1): " << result.sphericity << "\n";
        // std::cout << "Spread along minor axis s: " << result.spread_s << " m\n";
        // std::cout << "Eigenvalues: [" << result.eigenvalues.transpose() << "]\n";

        result.voxel_size = computeAdaptiveVoxelSize(result.sphericity, result.spread_s, params);
        // std::cout << "Adaptive voxel size v: " << result.voxel_size << " m\n";
    }

    pcl::VoxelGrid<PointT> voxel_filter;
    voxel_filter.setLeafSize(result.voxel_size, result.voxel_size, result.voxel_size);

    CloudPtr src_voxelized(new CloudT);
    CloudPtr tgt_voxelized(new CloudT);

    voxel_filter.setInputCloud(cloud_src);
    voxel_filter.filter(*src_voxelized);

    voxel_filter.setInputCloud(cloud_tgt);
    voxel_filter.filter(*tgt_voxelized);

    // std::cout << "After voxelization: Source " << src_voxelized->size() << ", Target "
    //           << tgt_voxelized->size() << " points\n";

    CloudPtr radius_estimation_cloud = selectLargerCloud(src_voxelized, tgt_voxelized);

    // std::cout << "\n=== Density-aware Radius Estimation ===\n";

    CloudPtr radius_samples;
    if (radius_estimation_cloud->size() > static_cast<size_t>(params.N_r)) {
        pcl::RandomSample<PointT> sampler;
        sampler.setInputCloud(radius_estimation_cloud);
        sampler.setSample(params.N_r);
        radius_samples.reset(new CloudT);
        sampler.filter(*radius_samples);
    } else {
        radius_samples = radius_estimation_cloud;
    }

    // std::cout << "Using " << radius_samples->size() << " points for radius estimation\n";

    const double scale_targets[3] = {params.tau_l, params.tau_m, params.tau_g};
    double* radius[3] = {&result.r_local, &result.r_middle, &result.r_global};
    const char* scale_names[3] = {"Local", "Middle", "Global"};

    #pragma omp parallel for
    for (int s = 0; s < 3; ++s) {
        const double r = estimateRadiusForDensity(radius_samples, scale_targets[s], params.r_max, 200);
        *(radius[s]) = r;

        #pragma omp critical
        {
            // std::cout << scale_names[s] << " scale (tau=" << scale_targets[s] << "): r = " << r
            //           << " m\n";
        }
    }

    if (result.r_local > result.r_middle || result.r_middle > result.r_global) {
        // std::cerr << "Warning: Radius hierarchy violated!\n";
        result.r_local = std::min(result.r_local, result.r_middle);
        result.r_middle = std::min(result.r_middle, result.r_global);
    }

    return result;
}

CloudPtr loadKittiPointCloud(const std::string& filepath) {
    return loadBinPointCloudXYZI(filepath, "KITTI");
}

CloudPtr loadMulranPointCloud(const std::string& filepath) {
    return loadBinPointCloudXYZI(filepath, "MulRan");
}

CloudPtr loadOxfordPointCloud(const std::string& filepath) {
    CloudPtr cloud(new CloudT);
    if (pcl::io::loadPCDFile<PointT>(filepath, *cloud) < 0) {
        std::cerr << "Error: Cannot open Oxford PCD file: " << filepath << "\n";
        return CloudPtr(new CloudT);
    }

    std::cout << "Loaded " << cloud->size() << " points from " << filepath << "\n";
    return cloud;
}

CloudPtr loadPointCloudByDataset(const std::string& filepath, const std::string& dataset) {
    if (dataset == "kitti" || dataset == "KITTI") {
        return loadKittiPointCloud(filepath);
    }
    if (dataset == "mulran" || dataset == "MulRan") {
        return loadMulranPointCloud(filepath);
    }
    if (dataset == "oxford" || dataset == "OXFORD") {
        return loadOxfordPointCloud(filepath);
    }
    throw std::invalid_argument("Unknown dataset: " + dataset);
}
