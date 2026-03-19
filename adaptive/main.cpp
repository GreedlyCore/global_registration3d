#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <fstream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/pca.h>
#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/random_sample.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <omp.h>

// Type aliases for convenience
using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using CloudPtr = CloudT::Ptr;

// Parameters structure matching the paper
struct GeometricBootstrappingParams {
    // Voxelization parameters
    double kappa_spheric = 0.10;   // Coefficient for high sphericity
    double kappa_disc = 0.15;      // Coefficient for low sphericity
    double tau_v = 0.05;           // Sphericity threshold
    double delta_v = 0.10;          // Sampling ratio for voxelization (10%)
    
    // Radius estimation parameters
    double tau_l = 0.005;           // Local density target
    double tau_m = 0.02;            // Middle density target
    double tau_g = 0.05;            // Global density target
    int N_r = 2000;                  // Points for radius estimation
    double r_max = 5.0;              // Maximum radius [m]
};

// Structure to hold results
struct GeometricBootstrappingResult {
    double voxel_size;               // Computed voxel size v
    double sphericity;                // λ₃/λ₁
    double spread_s;                  // Spread along minor axis
    Eigen::Vector3d eigenvalues;       // λ₁, λ₂, λ₃
    Eigen::Matrix3d eigenvectors;      // v₁, v₂, v₃
    
    // Multi-scale radius
    double r_local;
    double r_middle;
    double r_global;
};

// Function to select the larger point cloud based on cardinality
CloudPtr selectLargerCloud(const CloudPtr& cloud1, const CloudPtr& cloud2) {
    if (cloud1->size() >= cloud2->size()) {
        return cloud1;
    }
    return cloud2;
}

// Function to randomly sample percentage of points
CloudPtr samplePoints(const CloudPtr& cloud, double percentage) {
    if (percentage >= 1.0) return cloud;
    
    int sample_size = static_cast<int>(cloud->size() * percentage);
    if (sample_size <= 0) sample_size = 1;
    
    pcl::RandomSample<PointT> random_sample;
    random_sample.setInputCloud(cloud);
    random_sample.setSample(sample_size);
    
    CloudPtr sampled(new CloudT);
    random_sample.filter(*sampled);
    
    return sampled;
}

// Function to compute sphericity and related metrics via PCA
bool computeSphericityMetrics(const CloudPtr& cloud, 
                               double& sphericity,
                               double& spread_s,
                               Eigen::Vector3d& eigenvalues,
                               Eigen::Matrix3d& eigenvectors) {
    
    if (cloud->empty()) return false;
    
    // Compute centroid
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    
    // Build covariance matrix
    Eigen::Matrix3f cov;
    pcl::computeCovarianceMatrix(*cloud, centroid, cov);
    
    // Compute eigenvalues/vectors using Eigen
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
    if (solver.info() != Eigen::Success) return false;
    
    eigenvalues = solver.eigenvalues().cast<double>();
    eigenvectors = solver.eigenvectors().cast<double>();
    
    // Sort in descending order (λ₁ ≥ λ₂ ≥ λ₃)
    // Eigen's SelfAdjointEigenSolver returns in ascending order by default
    eigenvalues.reverseInPlace();
    
    // Reorder eigenvectors accordingly
    Eigen::Matrix3d eigvecs_asc = eigenvectors;
    eigenvectors.col(0) = eigvecs_asc.col(2);
    eigenvectors.col(1) = eigvecs_asc.col(1);
    eigenvectors.col(2) = eigvecs_asc.col(0);
    
    // Compute sphericity: λ₃/λ₁
    sphericity = eigenvalues(2) / eigenvalues(0);
    
    // Compute spread s along smallest eigenvector v₃
    Eigen::Vector3d v3 = eigenvectors.col(2);
    
    double min_proj = std::numeric_limits<double>::max();
    double max_proj = std::numeric_limits<double>::lowest();
    
    // Parallel projection computation using OpenMP
    #pragma omp parallel for reduction(min: min_proj) reduction(max: max_proj)
    for (int i = 0; i < static_cast<int>(cloud->size()); ++i) {
        const PointT& p = (*cloud)[i];
        Eigen::Vector3d point(p.x, p.y, p.z);
        double proj = point.dot(v3);
        
        if (proj < min_proj) min_proj = proj;
        if (proj > max_proj) max_proj = proj;
    }
    
    spread_s = max_proj - min_proj;
    
    return true;
}

// Function to determine adaptive voxel size
double computeAdaptiveVoxelSize(double sphericity, double spread_s,
                                 const GeometricBootstrappingParams& params) {
    
    double voxel_size;
    
    if (sphericity >= params.tau_v) {
        // Spherical point cloud (RGB-D style) - smaller voxels
        voxel_size = params.kappa_spheric * std::sqrt(spread_s);
    } else {
        // Disc-like point cloud (LiDAR style) - larger voxels
        voxel_size = params.kappa_disc * std::sqrt(spread_s);
    }
    
    return voxel_size;
}

// Function to estimate density-aware radius for a given target density
double estimateRadiusForDensity(const CloudPtr& cloud, 
                                 double target_density,
                                 double r_max,
                                 int num_queries) {
    
    if (cloud->empty()) return r_max;
    
    // Sample query points (could also use all points, but use subset for efficiency)
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
    
    // Build KD-tree for neighborhood search
    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);
    
    // Binary search for optimal radius
    double r_low = 0.01;      // 1cm minimum
    double r_high = r_max;
    double r_opt = r_high;
    double best_error = std::numeric_limits<double>::max();
    
    const int num_iterations = 20;  // Binary search iterations
    const double tolerance = 0.001;  // 1mm tolerance
    
    for (int iter = 0; iter < num_iterations; ++iter) {
        double r_mid = (r_low + r_high) * 0.5;
        
        // Compute average neighborhood size at radius r_mid
        double avg_neighbors = 0.0;
        
        #pragma omp parallel for reduction(+:avg_neighbors)
        for (int i = 0; i < static_cast<int>(query_points->size()); ++i) {
            PointT query = (*query_points)[i];
            
            std::vector<int> indices;
            std::vector<float> distances;
            
            if (kdtree.radiusSearch(query, r_mid, indices, distances) > 0) {
                avg_neighbors += static_cast<double>(indices.size());
            }
        }
        
        avg_neighbors /= static_cast<double>(query_points->size());
        
        // Compute error relative to target
        double error = std::abs(avg_neighbors - target_density * cloud->size());
        
        if (error < best_error) {
            best_error = error;
            r_opt = r_mid;
        }
        
        // Adjust binary search range
        if (avg_neighbors < target_density * cloud->size()) {
            r_low = r_mid;  // Need larger radius to include more points
        } else {
            r_high = r_mid; // Need smaller radius
        }
        
        if (r_high - r_low < tolerance) break;
    }
    
    return std::min(r_opt, r_max);
}

// Main pipeline function for geometric bootstrapping
GeometricBootstrappingResult geometricBootstrappingPipeline(
    const CloudPtr& cloud_src,
    const CloudPtr& cloud_tgt,
    const GeometricBootstrappingParams& params) {
    
    GeometricBootstrappingResult result;
    
    std::cout << "=== Geometric Bootstrapping Pipeline ===\n";
    
    // Step 1: Select the larger point cloud
    CloudPtr larger_cloud = selectLargerCloud(cloud_src, cloud_tgt);
    std::cout << "Selected larger cloud with " << larger_cloud->size() << " points\n";
    
    // Step 2: Sample points for sphericity analysis
    CloudPtr sampled_cloud = samplePoints(larger_cloud, params.delta_v);
    std::cout << "Sampled " << sampled_cloud->size() << " points for sphericity analysis\n";
    
    // Step 3: Compute sphericity metrics via PCA
    bool pca_success = computeSphericityMetrics(sampled_cloud, 
                                                 result.sphericity,
                                                 result.spread_s,
                                                 result.eigenvalues,
                                                 result.eigenvectors);
    
    if (!pca_success) {
        std::cerr << "PCA computation failed!\n";
        result.voxel_size = 0.1;  // Default fallback
    } else {
        std::cout << "Sphericity (λ₃/λ₁): " << result.sphericity << "\n";
        std::cout << "Spread along minor axis s: " << result.spread_s << " m\n";
        std::cout << "Eigenvalues: [" << result.eigenvalues.transpose() << "]\n";
        
        // Step 4: Determine adaptive voxel size
        result.voxel_size = computeAdaptiveVoxelSize(result.sphericity, 
                                                      result.spread_s, 
                                                      params);
        std::cout << "Adaptive voxel size v: " << result.voxel_size << " m\n";
    }
    
    // Step 5: Apply voxelization to both clouds (optional)
    pcl::VoxelGrid<PointT> voxel_filter;
    voxel_filter.setLeafSize(result.voxel_size, result.voxel_size, result.voxel_size);
    
    CloudPtr src_voxelized(new CloudT);
    CloudPtr tgt_voxelized(new CloudT);
    
    voxel_filter.setInputCloud(cloud_src);
    voxel_filter.filter(*src_voxelized);
    
    voxel_filter.setInputCloud(cloud_tgt);
    voxel_filter.filter(*tgt_voxelized);
    
    std::cout << "After voxelization: Source " << src_voxelized->size() 
              << ", Target " << tgt_voxelized->size() << " points\n";
    
    // Step 6: Select cloud for radius estimation (largest after voxelization)
    CloudPtr radius_estimation_cloud = selectLargerCloud(src_voxelized, tgt_voxelized);
    
    // Step 7: Density-aware radius estimation for three scales
    std::cout << "\n=== Density-aware Radius Estimation ===\n";
    
    // Sample points for radius estimation
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
    
    std::cout << "Using " << radius_samples->size() << " points for radius estimation\n";
    
    // Estimate radius for each scale
    double scale_targets[3] = {params.tau_l, params.tau_m, params.tau_g};
    double* radius[3] = {&result.r_local, &result.r_middle, &result.r_global};
    const char* scale_names[3] = {"Local", "Middle", "Global"};
    
    // Can parallelize scale estimation if independent TODO: check it out
    #pragma omp parallel for
    for (int s = 0; s < 3; ++s) {
        double r = estimateRadiusForDensity(radius_samples, 
                                             scale_targets[s],
                                             params.r_max,
                                             200);  // Use 200 query points
        
        *(radius[s]) = r;
        
        #pragma omp critical
        {
            std::cout << scale_names[s] << " scale (τ=" << scale_targets[s] 
                      << "): r = " << r << " m\n";
        }
    }
    
    // Verify hierarchy (should hold automatically from τ_l ≤ τ_m ≤ τ_g)
    if (result.r_local > result.r_middle || result.r_middle > result.r_global) {
        std::cerr << "Warning: Radius hierarchy violated!\n";
        // Enforce hierarchy as fallback
        result.r_local = std::min(result.r_local, result.r_middle);
        result.r_middle = std::min(result.r_middle, result.r_global);
    }
    
    return result;
}


// Load KITTI velodyne .bin file (x, y, z, reflectance as float32)
CloudPtr loadKittiPointCloud(const std::string& filepath) {
    CloudPtr cloud(new CloudT);
    
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open KITTI file: " << filepath << "\n";
        return cloud;
    }
    
    // Read floats (x, y, z, reflectance per point)
    float fx, fy, fz, reflectance;
    while (file.read(reinterpret_cast<char*>(&fx), sizeof(float))) {
        file.read(reinterpret_cast<char*>(&fy), sizeof(float));
        file.read(reinterpret_cast<char*>(&fz), sizeof(float));
        file.read(reinterpret_cast<char*>(&reflectance), sizeof(float));
        
        if (file.eof()) break;
        
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

// Example usage
int main() {
    const bool USE_KITTI_DEMO = true;
    
    CloudPtr cloud_src(new CloudT);
    CloudPtr cloud_tgt(new CloudT);
    
    if (USE_KITTI_DEMO) {
        // ===== KITTI DEMO =====
        std::cout << "\n=== KITTI Dataset Demo (Sequence 05) ===\n\n";
        
        std::string kitti_base = "/home/sonieth2/thesis/global_registration3d/data/KITTI/sequences/05/velodyne";
        std::string src_file = kitti_base + "/000027.bin";
        std::string tgt_file = kitti_base + "/000030.bin";
        
        cloud_src = loadKittiPointCloud(src_file);
        cloud_tgt = loadKittiPointCloud(tgt_file);
        
        if (cloud_src->empty() || cloud_tgt->empty()) {
            std::cerr << "Failed to load KITTI files. Check paths exist.\n";
            return 1;
        }
    } else {
        // ===== SYNTHETIC DEMO =====
        std::cout << "\n=== Synthetic LiDAR Demo (Disc-shaped clouds) ===\n\n";
    
        // Generate synthetic LiDAR-like disc-shaped cloud
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> xy_dist(-1.0, 1.0);
        std::uniform_real_distribution<> z_dist(-0.5, 0.5);  // Disc-like (small z variation)
        
        for (int i = 0; i < 100000; ++i) {
            PointT p;
            p.x = xy_dist(gen);
            p.y = xy_dist(gen);
            p.z = z_dist(gen);
            cloud_src->push_back(p);
        }
        
        for (int i = 0; i < 80000; ++i) {
            PointT p;
            p.x = xy_dist(gen) + 3.0;  // Slightly shifted
            p.y = xy_dist(gen) - 0.2;
            p.z = z_dist(gen);
            cloud_tgt->push_back(p);
        }
    }
    
    std::cout << "Source cloud: " << cloud_src->size() << " points\n";
    std::cout << "Target cloud: " << cloud_tgt->size() << " points\n";
    
    // Set parameters
    GeometricBootstrappingParams params;
    
    // Run pipeline
    GeometricBootstrappingResult result = geometricBootstrappingPipeline(cloud_src, cloud_tgt, params);
    
    // Output final results
    std::cout << "\n=== Final Results ===\n";
    std::cout << "Voxel size: " << result.voxel_size << " m\n";
    std::cout << "radius - Local: " << result.r_local 
              << " m, Middle: " << result.r_middle 
              << " m, Global: " << result.r_global << " m\n";
    
    return 0;
}
