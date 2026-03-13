/**
 * Exposes a single function:
 *   mac_solve(src_xyz, tgt_xyz, inlier_thresh, cmp_thresh, min_clique_size)
 *     -> (4, 4) float64 numpy array
 *
 * Build:
 *   mkdir build_pybind && cd build_pybind
 *   cmake -DCMAKE_BUILD_TYPE=Release -f ../CMakeLists_pybind.txt ..
 *   make -j$(nproc)
 */

// ------------------------------------------------------------------ //
// Global flags expected by Eva.h / registration.cpp / PCR.cpp
// (normally defined in main.cpp — we own them here instead)
// ------------------------------------------------------------------ //
bool add_overlap    = false;
bool low_inlieratio = false;
bool no_logs        = true;   // suppress all file I/O

// ------------------------------------------------------------------ //
// Standard includes
// ------------------------------------------------------------------ //
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Eigen>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "Eva.h"   // all MAC type definitions and function declarations

namespace py = pybind11;

// ------------------------------------------------------------------ //
// Build vector<Corre_3DMatch> from two (K, 3) float32 numpy arrays
// ------------------------------------------------------------------ //
static std::vector<Corre_3DMatch>
build_correspondences(py::array_t<float, py::array::c_style | py::array::forcecast> src_xyz,
                      py::array_t<float, py::array::c_style | py::array::forcecast> tgt_xyz)
{
    auto src = src_xyz.unchecked<2>();
    auto tgt = tgt_xyz.unchecked<2>();
    int K = static_cast<int>(src.shape(0));

    std::vector<Corre_3DMatch> corr(K);
    for (int i = 0; i < K; ++i) {
        corr[i].src.x = src(i, 0);
        corr[i].src.y = src(i, 1);
        corr[i].src.z = src(i, 2);
        corr[i].des.x = tgt(i, 0);
        corr[i].des.y = tgt(i, 1);
        corr[i].des.z = tgt(i, 2);
        corr[i].score        = 0.0;
        corr[i].inlier_weight = 0;
    }
    return corr;
}

// ------------------------------------------------------------------ //
// Helper: return identity (4,4) matrix as numpy array
// ------------------------------------------------------------------ //
static py::array_t<double> identity_result()
{
    py::array_t<double> result({4, 4});
    auto r = result.mutable_unchecked<2>();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            r(i, j) = (i == j) ? 1.0 : 0.0;
    return result;
}

// ------------------------------------------------------------------ //
// Main solver
// ------------------------------------------------------------------ //
py::array_t<double> mac_solve(
    py::array_t<float, py::array::c_style | py::array::forcecast> src_xyz,
    py::array_t<float, py::array::c_style | py::array::forcecast> tgt_xyz,
    float inlier_thresh,
    float cmp_thresh    = 0.99f,
    int   min_clique_sz = 3)
{
    // ---- correspondences ------------------------------------------ //
    if (src_xyz.shape(0) != tgt_xyz.shape(0) || src_xyz.shape(1) != 3)
        throw std::invalid_argument("src_xyz and tgt_xyz must both be (K, 3) float32 arrays");

    std::vector<Corre_3DMatch> correspondence = build_correspondences(src_xyz, tgt_xyz);
    int total_num = static_cast<int>(correspondence.size());
    if (total_num < min_clique_sz)
        return identity_result();

    // ---- graph construction --------------------------------------- //
    // Generic overload: score = exp(-dis^2 / (2 * alpha_dis^2)), alpha_dis = 10 * resolution
    // We set resolution = inlier_thresh / 10 so that alpha_dis == inlier_thresh.
    float resolution = inlier_thresh / 10.0f;
    bool  sc2        = true;

    Eigen::MatrixXf Graph =
        Graph_construction(correspondence, resolution, sc2, cmp_thresh);

    if (Graph.norm() == 0.0f)
        return identity_result();

    // ---- cluster coefficient computation ------------------------- //
    std::vector<int>      degree(total_num, 0);
    std::vector<Vote_exp> pts_degree;
    pts_degree.reserve(total_num);

    for (int i = 0; i < total_num; ++i) {
        Vote_exp t;
        t.true_num = 0;
        for (int j = 0; j < total_num; ++j) {
            if (i != j && Graph(i, j) != 0.0f) {
                degree[i]++;
                t.corre_index.push_back(j);
            }
        }
        t.index  = i;
        t.degree = degree[i];
        pts_degree.push_back(t);
    }

    std::vector<Vote> cluster_factor;
    cluster_factor.reserve(total_num);
    double sum_fenzi = 0.0, sum_fenmu = 0.0;

    for (int i = 0; i < total_num; ++i) {
        Vote   t;
        double wijk        = 0.0;
        int    index_size  = static_cast<int>(pts_degree[i].corre_index.size());

        for (int j = 0; j < index_size; ++j) {
            int a = pts_degree[i].corre_index[j];
            for (int k = j + 1; k < index_size; ++k) {
                int b = pts_degree[i].corre_index[k];
                if (Graph(a, b) != 0.0f)
                    wijk += std::pow(
                        static_cast<double>(Graph(i, a)) *
                        static_cast<double>(Graph(i, b)) *
                        static_cast<double>(Graph(a, b)), 1.0 / 3.0);
            }
        }

        t.index = i;
        if (degree[i] > 1) {
            double f2 = degree[i] * (degree[i] - 1) * 0.5;
            sum_fenzi += wijk;
            sum_fenmu += f2;
            t.score = wijk / f2;
        } else {
            t.score = 0.0;
        }
        cluster_factor.push_back(t);
    }

    double average_factor = 0.0;
    for (auto& cf : cluster_factor) average_factor += cf.score;
    average_factor /= static_cast<double>(cluster_factor.size());

    double total_factor = (sum_fenmu > 0.0) ? sum_fenzi / sum_fenmu : 0.0;

    std::vector<Vote> cluster_factor_bac = cluster_factor;
    std::sort(cluster_factor.begin(), cluster_factor.end(), compare_vote_score);
    std::sort(pts_degree.begin(),     pts_degree.end(),     compare_vote_degree);

    Eigen::VectorXd cluster_coefficients(static_cast<int>(cluster_factor.size()));
    for (int i = 0; i < static_cast<int>(cluster_factor.size()); ++i)
        cluster_coefficients[i] = cluster_factor[i].score;

    double OTSU = 0.0;
    if (cluster_factor[0].score != 0.0)
        OTSU = OTSU_thresh(cluster_coefficients);

    double cluster_threshold = std::min(OTSU, std::min(average_factor, total_factor));

    // assign cluster scores back (instance_equal = true, so all unit weight)
    for (int i = 0; i < total_num; ++i)
        correspondence[i].score = cluster_factor_bac[i].score;

    // ---- igraph setup -------------------------------------------- //
    igraph_t      g;
    igraph_matrix_t g_mat;
    igraph_matrix_init(&g_mat, Graph.rows(), Graph.cols());

    // Optional graph size reduction (mirrors registration.cpp)
    bool reduced = (cluster_threshold > 3.0 &&
                    static_cast<int>(correspondence.size()) > 50);
    if (reduced) {
        float f = 10.0f;
        double max_cf = std::max(OTSU, total_factor);
        // cluster_factor is sorted descending; need at least 50 entries
        double ref_score = (cluster_factor.size() > 50)
                           ? cluster_factor[49].score : 0.0;
        while (f * max_cf > ref_score && f > 0.05f)
            f -= 0.05f;

        for (int i = 0; i < Graph.rows(); ++i) {
            if (cluster_factor_bac[i].score > f * max_cf) {
                for (int j = i + 1; j < Graph.cols(); ++j) {
                    if (cluster_factor_bac[j].score > f * max_cf) {
                        MATRIX(g_mat, i, j) = Graph(i, j);
                        MATRIX(g_mat, j, i) = Graph(i, j);
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < Graph.rows(); ++i) {
            for (int j = i + 1; j < Graph.cols(); ++j) {
                if (Graph(i, j) != 0.0f) {
                    MATRIX(g_mat, i, j) = Graph(i, j);
                    MATRIX(g_mat, j, i) = Graph(i, j);
                }
            }
        }
    }

    igraph_set_attribute_table(&igraph_cattribute_table);

    int clique_num = 0;

#ifdef IGRAPH_VERSION_OLD
    // ---- igraph 0.9.9 -------------------------------------------- //
    igraph_vector_t weights_old;
    igraph_vector_init(&weights_old, Graph.rows() * (Graph.cols() - 1) / 2);

    igraph_weighted_adjacency(&g, &g_mat, IGRAPH_ADJ_UNDIRECTED, 0, 1);
    const char* att = "weight";
    EANV(&g, att, &weights_old);

    igraph_vector_ptr_t cliques;
    igraph_vector_ptr_init(&cliques, 0);
    igraph_maximal_cliques(&g, &cliques, min_clique_sz, 0);
    clique_num = static_cast<int>(igraph_vector_ptr_size(&cliques));

    igraph_destroy(&g);
    igraph_matrix_destroy(&g_mat);
    igraph_vector_destroy(&weights_old);
#else
    // ---- igraph 0.10.6 ------------------------------------------- //
    igraph_vector_t weight;
    igraph_vector_init(&weight, 0);
    igraph_weighted_adjacency(&g, &g_mat, IGRAPH_ADJ_UNDIRECTED,
                              &weight, IGRAPH_LOOPS_ONCE);

    igraph_vector_int_list_t cliques;
    igraph_vector_int_list_init(&cliques, 0);
    igraph_maximal_cliques(&g, &cliques, min_clique_sz, 0);
    clique_num = static_cast<int>(igraph_vector_int_list_size(&cliques));

    igraph_destroy(&g);
    igraph_matrix_destroy(&g_mat);
    igraph_vector_destroy(&weight);
#endif

    if (clique_num == 0) {
#ifdef IGRAPH_VERSION_OLD
        igraph_vector_ptr_destroy(&cliques);
#else
        igraph_vector_int_list_destroy(&cliques);
#endif
        return identity_result();
    }

    // ---- clique selection ---------------------------------------- //
    std::vector<int> remain;
    remain.reserve(clique_num);
    for (int i = 0; i < clique_num; ++i) remain.push_back(i);

    node_cliques* N_C = new node_cliques[total_num];
    find_largest_clique_of_node(Graph, &cliques, correspondence,
                                N_C, remain, total_num, INT_MAX, "fpfh");
    delete[] N_C;

    // ---- build pcl clouds for all correspondences ---------------- //
    PointCloudPtr src_corr_pts(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr des_corr_pts(new pcl::PointCloud<pcl::PointXYZ>);
    src_corr_pts->reserve(total_num);
    des_corr_pts->reserve(total_num);
    for (auto& c : correspondence) {
        src_corr_pts->push_back(c.src);
        des_corr_pts->push_back(c.des);
    }

    // ---- hypothesis evaluation (parallel with OMP) --------------- //
    double          best_score = 0.0;
    Eigen::Matrix4d best_est   = Eigen::Matrix4d::Identity();

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(remain.size()); ++i) {
        std::vector<Corre_3DMatch> Group;

#ifdef IGRAPH_VERSION_OLD
        igraph_vector_t* v =
            reinterpret_cast<igraph_vector_t*>(VECTOR(cliques)[remain[i]]);
        int gsz = static_cast<int>(igraph_vector_size(v));
        for (int j = 0; j < gsz; ++j)
            Group.push_back(correspondence[static_cast<int>(VECTOR(*v)[j])]);
#else
        igraph_vector_int_t* v =
            igraph_vector_int_list_get_ptr(&cliques, remain[i]);
        int gsz = static_cast<int>(igraph_vector_int_size(v));
        for (int j = 0; j < gsz; ++j)
            Group.push_back(correspondence[VECTOR(*v)[j]]);
#endif

        Eigen::Matrix4d est_trans;
        double score = evaluation_trans(
            Group, correspondence,
            src_corr_pts, des_corr_pts,
            /*weight_thresh=*/0.0,
            est_trans,
            /*metric_thresh=*/static_cast<double>(inlier_thresh),
            /*metric=*/"MAE",
            /*resolution=*/0.0f,
            /*instance_equal=*/true);

#pragma omp critical
        {
            if (score > best_score) {
                best_score = score;
                best_est   = est_trans;
            }
        }
    }

    // ---- free clique memory -------------------------------------- //
#ifdef IGRAPH_VERSION_OLD
    igraph_vector_ptr_destroy(&cliques);
#else
    igraph_vector_int_list_destroy(&cliques);
#endif

    // ---- post refinement ----------------------------------------- //
    if (best_score > 0.0) {
        post_refinement(correspondence, src_corr_pts, des_corr_pts,
                        best_est, best_score,
                        static_cast<double>(inlier_thresh),
                        /*iterations=*/20, "MAE");
    }

    // ---- pack result --------------------------------------------- //
    py::array_t<double> result({4, 4});
    auto r = result.mutable_unchecked<2>();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            r(i, j) = best_est(i, j);
    return result;
}

// ------------------------------------------------------------------ //
// Verbose solver — same logic as mac_solve but also returns graph-state
// metrics needed for the parameter sweep analysis.
// ------------------------------------------------------------------ //
py::dict mac_solve_verbose(
    py::array_t<float, py::array::c_style | py::array::forcecast> src_xyz,
    py::array_t<float, py::array::c_style | py::array::forcecast> tgt_xyz,
    float inlier_thresh,
    float cmp_thresh    = 0.99f,
    int   min_clique_sz = 3,
    py::object is_inlier_arg = py::none())   // optional (K,) bool array
{
    const double kNaN = std::numeric_limits<double>::quiet_NaN();

    // Helper lambda: pack a "no-solution" dict
    auto empty_dict = [&]() -> py::dict {
        py::dict d;
        d["transform"]          = identity_result();
        d["n_cliques_total"]    = 0;
        d["n_cliques_selected"] = 0;
        d["n_edges"]            = 0;
        d["graph_density"]      = kNaN;
        d["mean_deg_inlier"]    = kNaN;
        d["mean_deg_outlier"]   = kNaN;
        d["sep"]                = kNaN;
        d["f_pure"]             = kNaN;
        d["r_star"]             = -1;
        return d;
    };

    // ---- correspondences ------------------------------------------ //
    if (src_xyz.shape(0) != tgt_xyz.shape(0) || src_xyz.shape(1) != 3)
        throw std::invalid_argument("src_xyz and tgt_xyz must both be (K, 3) float32 arrays");

    std::vector<Corre_3DMatch> correspondence = build_correspondences(src_xyz, tgt_xyz);
    int total_num = static_cast<int>(correspondence.size());
    if (total_num < min_clique_sz)
        return empty_dict();

    // ---- optional GT inlier mask ---------------------------------- //
    std::vector<bool> is_inlier_vec;
    bool have_labels = !is_inlier_arg.is_none();
    if (have_labels) {
        auto arr = py::cast<py::array_t<bool, py::array::c_style | py::array::forcecast>>(
            is_inlier_arg);
        if (arr.shape(0) != total_num)
            throw std::invalid_argument("is_inlier must have length K (same as correspondences)");
        auto buf = arr.unchecked<1>();
        is_inlier_vec.resize(total_num);
        for (int i = 0; i < total_num; ++i)
            is_inlier_vec[i] = buf(i);
    }

    // ---- graph construction --------------------------------------- //
    float resolution = inlier_thresh / 10.0f;
    bool  sc2        = true;

    Eigen::MatrixXf Graph =
        Graph_construction(correspondence, resolution, sc2, cmp_thresh);

    if (Graph.norm() == 0.0f)
        return empty_dict();

    // ---- degree computation + graph density ---------------------- //
    std::vector<int> degree(total_num, 0);
    int n_edges = 0;
    for (int i = 0; i < total_num; ++i)
        for (int j = i + 1; j < total_num; ++j)
            if (Graph(i, j) != 0.0f) {
                degree[i]++;
                degree[j]++;
                n_edges++;
            }

    double graph_density = kNaN;
    if (total_num > 1)
        graph_density = 2.0 * n_edges / (static_cast<double>(total_num) * (total_num - 1));

    // ---- degree stats per inlier/outlier label ------------------- //
    double mean_deg_inlier = kNaN, mean_deg_outlier = kNaN, sep = kNaN;
    if (have_labels) {
        double sum_in = 0.0, sum_out = 0.0;
        int    cnt_in = 0,   cnt_out = 0;
        for (int i = 0; i < total_num; ++i) {
            if (is_inlier_vec[i]) { sum_in  += degree[i]; cnt_in++;  }
            else                  { sum_out += degree[i]; cnt_out++; }
        }
        if (cnt_in  > 0) mean_deg_inlier  = sum_in  / cnt_in;
        if (cnt_out > 0) mean_deg_outlier = sum_out / cnt_out;
        if (cnt_out > 0 && mean_deg_outlier > 0.0)
            sep = mean_deg_inlier / mean_deg_outlier;
        else if (cnt_out == 0)
            sep = kNaN;  // no outliers — undefined
    }

    // ---- cluster coefficient computation (unchanged from mac_solve) //
    std::vector<Vote_exp> pts_degree;
    pts_degree.reserve(total_num);

    for (int i = 0; i < total_num; ++i) {
        Vote_exp t;
        t.true_num = 0;
        for (int j = 0; j < total_num; ++j)
            if (i != j && Graph(i, j) != 0.0f)
                t.corre_index.push_back(j);
        t.index  = i;
        t.degree = degree[i];
        pts_degree.push_back(t);
    }

    std::vector<Vote> cluster_factor;
    cluster_factor.reserve(total_num);
    double sum_fenzi = 0.0, sum_fenmu = 0.0;

    for (int i = 0; i < total_num; ++i) {
        Vote   t;
        double wijk       = 0.0;
        int    index_size = static_cast<int>(pts_degree[i].corre_index.size());

        for (int j = 0; j < index_size; ++j) {
            int a = pts_degree[i].corre_index[j];
            for (int k = j + 1; k < index_size; ++k) {
                int b = pts_degree[i].corre_index[k];
                if (Graph(a, b) != 0.0f)
                    wijk += std::pow(
                        static_cast<double>(Graph(i, a)) *
                        static_cast<double>(Graph(i, b)) *
                        static_cast<double>(Graph(a, b)), 1.0 / 3.0);
            }
        }

        t.index = i;
        if (degree[i] > 1) {
            double f2 = degree[i] * (degree[i] - 1) * 0.5;
            sum_fenzi += wijk;
            sum_fenmu += f2;
            t.score = wijk / f2;
        } else {
            t.score = 0.0;
        }
        cluster_factor.push_back(t);
    }

    double average_factor = 0.0;
    for (auto& cf : cluster_factor) average_factor += cf.score;
    average_factor /= static_cast<double>(cluster_factor.size());
    double total_factor = (sum_fenmu > 0.0) ? sum_fenzi / sum_fenmu : 0.0;

    std::vector<Vote> cluster_factor_bac = cluster_factor;
    std::sort(cluster_factor.begin(), cluster_factor.end(), compare_vote_score);
    std::sort(pts_degree.begin(),     pts_degree.end(),     compare_vote_degree);

    Eigen::VectorXd cluster_coefficients(static_cast<int>(cluster_factor.size()));
    for (int i = 0; i < static_cast<int>(cluster_factor.size()); ++i)
        cluster_coefficients[i] = cluster_factor[i].score;

    double OTSU = 0.0;
    if (cluster_factor[0].score != 0.0)
        OTSU = OTSU_thresh(cluster_coefficients);

    double cluster_threshold = std::min(OTSU, std::min(average_factor, total_factor));

    for (int i = 0; i < total_num; ++i)
        correspondence[i].score = cluster_factor_bac[i].score;

    // ---- igraph setup -------------------------------------------- //
    igraph_t       g;
    igraph_matrix_t g_mat;
    igraph_matrix_init(&g_mat, Graph.rows(), Graph.cols());

    bool reduced = (cluster_threshold > 3.0 &&
                    static_cast<int>(correspondence.size()) > 50);
    if (reduced) {
        float  f       = 10.0f;
        double max_cf  = std::max(OTSU, total_factor);
        double ref_score = (cluster_factor.size() > 50)
                           ? cluster_factor[49].score : 0.0;
        while (f * max_cf > ref_score && f > 0.05f)
            f -= 0.05f;
        for (int i = 0; i < Graph.rows(); ++i)
            if (cluster_factor_bac[i].score > f * max_cf)
                for (int j = i + 1; j < Graph.cols(); ++j)
                    if (cluster_factor_bac[j].score > f * max_cf) {
                        MATRIX(g_mat, i, j) = Graph(i, j);
                        MATRIX(g_mat, j, i) = Graph(i, j);
                    }
    } else {
        for (int i = 0; i < Graph.rows(); ++i)
            for (int j = i + 1; j < Graph.cols(); ++j)
                if (Graph(i, j) != 0.0f) {
                    MATRIX(g_mat, i, j) = Graph(i, j);
                    MATRIX(g_mat, j, i) = Graph(i, j);
                }
    }

    igraph_set_attribute_table(&igraph_cattribute_table);
    int clique_num = 0;

#ifdef IGRAPH_VERSION_OLD
    igraph_vector_t weights_old;
    igraph_vector_init(&weights_old, Graph.rows() * (Graph.cols() - 1) / 2);
    igraph_weighted_adjacency(&g, &g_mat, IGRAPH_ADJ_UNDIRECTED, 0, 1);
    const char* att = "weight";
    EANV(&g, att, &weights_old);

    igraph_vector_ptr_t cliques;
    igraph_vector_ptr_init(&cliques, 0);
    igraph_maximal_cliques(&g, &cliques, min_clique_sz, 0);
    clique_num = static_cast<int>(igraph_vector_ptr_size(&cliques));

    igraph_destroy(&g);
    igraph_matrix_destroy(&g_mat);
    igraph_vector_destroy(&weights_old);
#else
    igraph_vector_t weight;
    igraph_vector_init(&weight, 0);
    igraph_weighted_adjacency(&g, &g_mat, IGRAPH_ADJ_UNDIRECTED,
                              &weight, IGRAPH_LOOPS_ONCE);

    igraph_vector_int_list_t cliques;
    igraph_vector_int_list_init(&cliques, 0);
    igraph_maximal_cliques(&g, &cliques, min_clique_sz, 0);
    clique_num = static_cast<int>(igraph_vector_int_list_size(&cliques));

    igraph_destroy(&g);
    igraph_matrix_destroy(&g_mat);
    igraph_vector_destroy(&weight);
#endif

    if (clique_num == 0) {
#ifdef IGRAPH_VERSION_OLD
        igraph_vector_ptr_destroy(&cliques);
#else
        igraph_vector_int_list_destroy(&cliques);
#endif
        return empty_dict();
    }

    // ---- clique selection ---------------------------------------- //
    std::vector<int> remain;
    remain.reserve(clique_num);
    for (int i = 0; i < clique_num; ++i) remain.push_back(i);

    node_cliques* N_C = new node_cliques[total_num];
    find_largest_clique_of_node(Graph, &cliques, correspondence,
                                N_C, remain, total_num, INT_MAX, "fpfh");
    delete[] N_C;

    int n_cliques_selected = static_cast<int>(remain.size());

    // ---- per-clique weights (for r* ranking) --------------------- //
    // weight(C) = sum of Graph(a,b) for all edges (a,b) in clique C
    std::vector<double> clique_weights(n_cliques_selected, 0.0);
    for (int i = 0; i < n_cliques_selected; ++i) {
#ifdef IGRAPH_VERSION_OLD
        igraph_vector_t* v =
            reinterpret_cast<igraph_vector_t*>(VECTOR(cliques)[remain[i]]);
        int gsz = static_cast<int>(igraph_vector_size(v));
        for (int j = 0; j < gsz; ++j)
            for (int k = j + 1; k < gsz; ++k)
                clique_weights[i] += Graph(
                    static_cast<int>(VECTOR(*v)[j]),
                    static_cast<int>(VECTOR(*v)[k]));
#else
        igraph_vector_int_t* v =
            igraph_vector_int_list_get_ptr(&cliques, remain[i]);
        int gsz = static_cast<int>(igraph_vector_int_size(v));
        for (int j = 0; j < gsz; ++j)
            for (int k = j + 1; k < gsz; ++k)
                clique_weights[i] += Graph(VECTOR(*v)[j], VECTOR(*v)[k]);
#endif
    }

    // Weight-sorted ranking (descending): rank[i] = 1-indexed position in sorted order
    std::vector<int> weight_rank_order(n_cliques_selected);
    std::iota(weight_rank_order.begin(), weight_rank_order.end(), 0);
    std::sort(weight_rank_order.begin(), weight_rank_order.end(),
              [&](int a, int b){ return clique_weights[a] > clique_weights[b]; });
    // weight_rank[i] = 1-based rank of remain-index i by clique weight
    std::vector<int> weight_rank(n_cliques_selected);
    for (int r = 0; r < n_cliques_selected; ++r)
        weight_rank[weight_rank_order[r]] = r + 1;

    // ---- build pcl clouds ---------------------------------------- //
    PointCloudPtr src_corr_pts(new pcl::PointCloud<pcl::PointXYZ>);
    PointCloudPtr des_corr_pts(new pcl::PointCloud<pcl::PointXYZ>);
    src_corr_pts->reserve(total_num);
    des_corr_pts->reserve(total_num);
    for (auto& c : correspondence) {
        src_corr_pts->push_back(c.src);
        des_corr_pts->push_back(c.des);
    }

    // ---- hypothesis evaluation ----------------------------------- //
    double          best_score = 0.0;
    Eigen::Matrix4d best_est   = Eigen::Matrix4d::Identity();
    int             best_remain_idx = -1;   // position in remain[]

#pragma omp parallel for
    for (int i = 0; i < n_cliques_selected; ++i) {
        std::vector<Corre_3DMatch> Group;

#ifdef IGRAPH_VERSION_OLD
        igraph_vector_t* v =
            reinterpret_cast<igraph_vector_t*>(VECTOR(cliques)[remain[i]]);
        int gsz = static_cast<int>(igraph_vector_size(v));
        for (int j = 0; j < gsz; ++j)
            Group.push_back(correspondence[static_cast<int>(VECTOR(*v)[j])]);
#else
        igraph_vector_int_t* v =
            igraph_vector_int_list_get_ptr(&cliques, remain[i]);
        int gsz = static_cast<int>(igraph_vector_int_size(v));
        for (int j = 0; j < gsz; ++j)
            Group.push_back(correspondence[VECTOR(*v)[j]]);
#endif

        Eigen::Matrix4d est_trans;
        double score = evaluation_trans(
            Group, correspondence,
            src_corr_pts, des_corr_pts,
            /*weight_thresh=*/0.0,
            est_trans,
            /*metric_thresh=*/static_cast<double>(inlier_thresh),
            /*metric=*/"MAE",
            /*resolution=*/0.0f,
            /*instance_equal=*/true);

#pragma omp critical
        {
            if (score > best_score) {
                best_score      = score;
                best_est        = est_trans;
                best_remain_idx = i;
            }
        }
    }

    // ---- r* ------------------------------------------------------- //
    int r_star = (best_remain_idx >= 0) ? weight_rank[best_remain_idx] : -1;

    // ---- f_pure ---------------------------------------------------- //
    double f_pure = kNaN;
    if (have_labels && n_cliques_selected > 0) {
        int pure_count = 0;
        for (int i = 0; i < n_cliques_selected; ++i) {
            bool all_inliers = true;
#ifdef IGRAPH_VERSION_OLD
            igraph_vector_t* v =
                reinterpret_cast<igraph_vector_t*>(VECTOR(cliques)[remain[i]]);
            int gsz = static_cast<int>(igraph_vector_size(v));
            for (int j = 0; j < gsz && all_inliers; ++j)
                if (!is_inlier_vec[static_cast<int>(VECTOR(*v)[j])])
                    all_inliers = false;
#else
            igraph_vector_int_t* v =
                igraph_vector_int_list_get_ptr(&cliques, remain[i]);
            int gsz = static_cast<int>(igraph_vector_int_size(v));
            for (int j = 0; j < gsz && all_inliers; ++j)
                if (!is_inlier_vec[VECTOR(*v)[j]])
                    all_inliers = false;
#endif
            if (all_inliers) pure_count++;
        }
        f_pure = static_cast<double>(pure_count) / n_cliques_selected;
    }

    // ---- free clique memory -------------------------------------- //
#ifdef IGRAPH_VERSION_OLD
    igraph_vector_ptr_destroy(&cliques);
#else
    igraph_vector_int_list_destroy(&cliques);
#endif

    // ---- post refinement ----------------------------------------- //
    if (best_score > 0.0) {
        post_refinement(correspondence, src_corr_pts, des_corr_pts,
                        best_est, best_score,
                        static_cast<double>(inlier_thresh),
                        /*iterations=*/20, "MAE");
    }

    // ---- pack transform ------------------------------------------ //
    py::array_t<double> transform({4, 4});
    auto r = transform.mutable_unchecked<2>();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            r(i, j) = best_est(i, j);

    // ---- return dict --------------------------------------------- //
    py::dict d;
    d["transform"]          = transform;
    d["n_cliques_total"]    = clique_num;
    d["n_cliques_selected"] = n_cliques_selected;
    d["n_edges"]            = n_edges;
    d["graph_density"]      = graph_density;
    d["mean_deg_inlier"]    = mean_deg_inlier;
    d["mean_deg_outlier"]   = mean_deg_outlier;
    d["sep"]                = sep;
    d["f_pure"]             = f_pure;
    d["r_star"]             = r_star;
    return d;
}

// ------------------------------------------------------------------ //
// Module
// ------------------------------------------------------------------ //
PYBIND11_MODULE(mac_solver, m)
{
    m.doc() = "MAC: 3D Registration with Maximal Cliques — C++ pybind11 binding";

    m.def("mac_solve", &mac_solve,
          py::arg("src_xyz"),
          py::arg("tgt_xyz"),
          py::arg("inlier_thresh"),
          py::arg("cmp_thresh")     = 0.99f,
          py::arg("min_clique_sz")  = 3,
          R"doc(
Run MAC registration solver on pre-computed correspondences.

Args:
    src_xyz       : (K, 3) float32 numpy array — source keypoint positions
    tgt_xyz       : (K, 3) float32 numpy array — target keypoint positions
    inlier_thresh : inlier distance threshold
                    (e.g. voxel_size for KITTI-scale data, 0.1 for 3DMatch)
    cmp_thresh    : compatibility graph edge keep threshold (default 0.99)
    min_clique_sz : minimum clique size for maximal clique search
                    (default 3; use 4 for KITTI)

Returns:
    (4, 4) float64 numpy array — estimated src->tgt transformation.
    Returns identity if no valid cliques are found.
)doc");

    m.def("mac_solve_verbose", &mac_solve_verbose,
          py::arg("src_xyz"),
          py::arg("tgt_xyz"),
          py::arg("inlier_thresh"),
          py::arg("cmp_thresh")     = 0.99f,
          py::arg("min_clique_sz")  = 3,
          py::arg("is_inlier")      = py::none(),
          R"doc(
Run MAC registration solver and return graph-state metrics alongside the transform.

Args:
    src_xyz       : (K, 3) float32 numpy array — source keypoint positions
    tgt_xyz       : (K, 3) float32 numpy array — target keypoint positions
    inlier_thresh : inlier distance threshold
    cmp_thresh    : compatibility graph edge keep threshold (default 0.99)
    min_clique_sz : minimum clique size (default 3)
    is_inlier     : optional (K,) bool numpy array — GT inlier mask per correspondence.
                    When provided, enables mean_deg_inlier, mean_deg_outlier, sep, f_pure.

Returns:
    dict with keys:
      transform          : (4, 4) float64 — estimated src->tgt transformation
      n_cliques_total    : int   — total maximal cliques found by igraph
      n_cliques_selected : int   — cliques remaining after node-guided selection
      n_edges            : int   — edges in the compatibility graph
      graph_density      : float — 2|E| / (|V|(|V|-1))
      mean_deg_inlier    : float — mean node degree of GT inlier correspondences (nan if no labels)
      mean_deg_outlier   : float — mean node degree of GT outlier correspondences (nan if no labels)
      sep                : float — mean_deg_inlier / mean_deg_outlier              (nan if no labels)
      f_pure             : float — fraction of selected cliques that are 100% inlier (nan if no labels)
      r_star             : int   — 1-indexed rank (by clique edge-weight sum) of the winning clique;
                                   -1 if no valid solution found
)doc");
}
