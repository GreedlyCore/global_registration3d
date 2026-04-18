#include <iostream>
#include <random>
#include <string>

#include "adaptive_lib.h"

int main(int argc, char** argv) {
    std::string demo_type = "kitti";
    if (argc > 1) {
        demo_type = argv[1];
    }

    CloudPtr cloud_src(new CloudT);
    CloudPtr cloud_tgt(new CloudT);

    if (demo_type == "kitti") {
        std::cout << "\n=== KITTI Dataset Demo (Sequence 05) ===\n\n";

        const std::string kitti_base = "/home/sonieth2/thesis/global_registration3d/data/KITTI/sequences/05/velodyne";
        const std::string src_file = kitti_base + "/000027.bin";
        const std::string tgt_file = kitti_base + "/000030.bin";

        cloud_src = loadKittiPointCloud(src_file);
        cloud_tgt = loadKittiPointCloud(tgt_file);

        if (cloud_src->empty() || cloud_tgt->empty()) {
            std::cerr << "Failed to load KITTI files. Check paths exist.\n";
            return 1;
        }
    } else if (demo_type == "oxford") {
        const std::string oxford_seq = "2024-03-20-christ-church-06";
        std::cout << "\n=== Oxford Dataset Demo (Sequence " << oxford_seq << ") ===\n\n";

        const std::string oxford_base =
            "/home/sonieth2/thesis/global_registration3d/data/OXFORD/" + oxford_seq + "/lidar-clouds";
        const std::string src_file = oxford_base + "/1710927714.179242000.pcd";
        const std::string tgt_file = oxford_base + "/1710927714.279141000.pcd";

        std::cout << "Sequence " << oxford_seq << " -> 1710927714.179242000.pcd\n";
        std::cout << "Sequence " << oxford_seq << " -> 1710927714.279141000.pcd\n";

        cloud_src = loadOxfordPointCloud(src_file);
        cloud_tgt = loadOxfordPointCloud(tgt_file);

        if (cloud_src->empty() || cloud_tgt->empty()) {
            std::cerr << "Failed to load Oxford files. Check paths exist.\n";
            return 1;
        }
    } else if (demo_type == "mulran") {
        const std::string mulran_seq = "RIVERSIDE02";
        std::cout << "\n=== MulRan Dataset Demo (Sequence " << mulran_seq << ") ===\n\n";

        const std::string mulran_base =
            "/home/sonieth2/thesis/global_registration3d/data/MulRan/" + mulran_seq + "/Ouster";
        const std::string src_file = mulran_base + "/1565943695327979143.bin";
        const std::string tgt_file = mulran_base + "/1565943695427969272.bin";

        std::cout << "Sequence " << mulran_seq << " -> 1565943695327979143.bin\n";
        std::cout << "Sequence " << mulran_seq << " -> 1565943695427969272.bin\n";

        cloud_src = loadMulranPointCloud(src_file);
        cloud_tgt = loadMulranPointCloud(tgt_file);

        if (cloud_src->empty() || cloud_tgt->empty()) {
            std::cerr << "Failed to load MulRan files. Check paths exist.\n";
            return 1;
        }
    } else if (demo_type == "simple") {
        std::cout << "\n=== Synthetic LiDAR Demo (Disc-shaped clouds) ===\n\n";

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> xy_dist(-1.0, 1.0);
        std::uniform_real_distribution<> z_dist(-0.5, 0.5);

        for (int i = 0; i < 100000; ++i) {
            PointT p;
            p.x = xy_dist(gen);
            p.y = xy_dist(gen);
            p.z = z_dist(gen);
            cloud_src->push_back(p);
        }

        for (int i = 0; i < 80000; ++i) {
            PointT p;
            p.x = xy_dist(gen) + 3.0;
            p.y = xy_dist(gen) - 0.2;
            p.z = z_dist(gen);
            cloud_tgt->push_back(p);
        }
    } else {
        std::cerr << "Unknown demo type: " << demo_type << "\n";
        std::cerr << "Use one of: simple, oxford, mulran, kitti\n";
        return 1;
    }

    std::cout << "Source cloud: " << cloud_src->size() << " points\n";
    std::cout << "Target cloud: " << cloud_tgt->size() << " points\n";

    GeometricBootstrappingParams params;
    GeometricBootstrappingResult result = geometricBootstrappingPipeline(cloud_src, cloud_tgt, params);

    std::cout << "\n=== Final Results ===\n";
    std::cout << "Voxel size: " << result.voxel_size << " m\n";
    std::cout << "radius - Local: " << result.r_local << " m, Middle: " << result.r_middle
              << " m, Global: " << result.r_global << " m\n";

    return 0;
}
