//
// Created by 谢哲 on 2021/8/7.
//

#ifndef GESCPP_GES_H
#define GESCPP_GES_H
#include <algorithm>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "DecomposableScore.h"
#include "torch/torch.h"
#include "utils.h"

namespace ges {
using ull = unsigned long long;

auto insert(int x, int y, const std::set<int>& T, const torch::Tensor& A) {
    auto new_A = A.clone();
    std::vector<int> T_vec{T.begin(), T.end()};
    auto T_torch = torch::tensor(T_vec);
    new_A[x][y] = 1;
    new_A.index_put_({T_torch, y}, 1);
    new_A.index_put_({y, T_torch}, 0);

    return new_A;
}

auto delete_node(int x, int y, const std::set<int>& H, const torch::Tensor& A) {
    auto new_A = A.clone();
    new_A[x][y] = 0;
    new_A[y][x] = 0;
    auto H_torch = torch::tensor(std::vector<int>{H.begin(), H.end()});
    new_A.index_put_({H_torch, y}, 0);
    auto n_x = utils::neighbors(x, A);
    std::vector<int> h_nx;
    std::set_intersection(H.begin(), H.end(), n_x.begin(), n_x.end(),
                          std::back_inserter(h_nx));
    auto h_nx_torch = torch::tensor(h_nx);
    new_A.index_put_({h_nx_torch, x}, 0);

    return new_A;
}

auto score_valid_insert_operators(int x,
                                  int y,
                                  const torch::Tensor& A,
                                  DecomposableScore& cache,
                                  int debug = 0) {
    int p = A.size(0);
    auto s1 = utils::neighbors(y, A), s2 = utils::adj(x, A);
    std::vector<int> T0;
    std::unordered_map<int, int> reversed_T0;

    std::set_difference(s1.begin(), s1.end(), s2.begin(), s2.end(),
                        std::back_inserter(T0));
    for (int i = 0; i < T0.size(); ++i)
        reversed_T0[T0[i]] = i;

    ull total_valid = (1 << T0.size());
    std::vector<bool> removed(total_valid, false);
    std::vector<bool> passed_cond_2(total_valid, false);

    int valid_count = 0, best_x = 0, best_y = 0;
    std::set<int> best_T;
    double best_score = -1e10;
    torch::Tensor best_A;

    // Traverse all subsets of T0
    for (ull sub = 0; sub < total_valid; ++sub) {
        if (removed[sub]) continue;
        // Check Cond 1
        std::set<int> T;
        for (int i = 0; i < T0.size(); ++i)
            if (((1ull << i) & sub) == (1ull << i)) T.insert(T0[i]);

        auto yxT = utils::na(y, x, A);
        std::set<int> na_yxT;
        set_union(yxT.begin(), yxT.end(), T.begin(), T.end(),
                  std::inserter(na_yxT, na_yxT.end()));
        auto cond_1 = utils::is_clique(na_yxT, A);

        if (!cond_1) {
            // Remove from other which contain T
            for (ull sup = 0; sup < total_valid; ++sup)
                if ((sup & sub) == sub) removed[sup] = true;
        }
        bool cond_2;
        if (passed_cond_2[sub]) {
            cond_2 = true;
        } else {
            cond_2 = true;
            for (const auto& path : utils::semi_directed_paths(y, x, A)) {
                int cnt = 0;
                for (auto node : path) {
                    if (na_yxT.find(node) != na_yxT.end()) ++cnt;
                }
                if (cnt == 0) {
                    cond_2 = false;
                    break;
                }
            }
            if (cond_2) {
                for (ull sup = 0; sup < total_valid; ++sup)
                    if ((sup & sub) == sub) passed_cond_2[sup] = true;
            }
        }
        if (cond_1 and cond_2) {
            auto new_A = insert(x, y, T, A);
            std::set<int> aux;
            auto pa_y = utils::pa(y, A);
            set_union(na_yxT.begin(), na_yxT.end(), pa_y.begin(), pa_y.end(),
                      std::inserter(aux, aux.end()));
            // Compute the change in score
            auto old_score = cache.local_score(y, aux);
            aux.insert(x);
            auto new_score = cache.local_score(y, aux);
            if (std::isinf(old_score) or std::isinf(new_score)) continue;
            if (debug) std::cout << new_score - old_score << std::endl;

            valid_count++;
            if (new_score - old_score > best_score) {
                best_score = new_score - old_score;
                best_A = new_A;
                best_x = x, best_y = y;
                best_T = T;
            }
        }
    }

    return std::make_tuple(best_score, best_A, valid_count, best_x, best_y,
                           best_T);
}

auto score_valid_delete_operators(int x,
                                  int y,
                                  const torch::Tensor& A,
                                  DecomposableScore& cache,
                                  int debug = 0) {
    auto na_yx = utils::na(y, x, A);
    std::vector<int> H0{na_yx.begin(), na_yx.end()};

    ull total_valid = (1 << H0.size());
    std::vector<bool> cond_1_list(total_valid, false);

    int valid_count = 0, best_x = 0, best_y = 0;
    double best_score = -1e10;
    std::set<int> best_T;
    torch::Tensor best_A;

    // Traverse all subsets of T0
    for (ull sub = 0; sub < total_valid; ++sub) {
        // Check Cond 1
        std::set<int> H;
        for (int i = 0; i < H0.size(); ++i)
            if (((1ull << i) & sub) == (1ull << i)) H.insert(H0[i]);

        // Check cond1
        auto cond_1 = cond_1_list[sub];
        std::set<int> na_yx_h;
        std::set_difference(na_yx.begin(), na_yx.end(), H.begin(), H.end(),
                            std::inserter(na_yx_h, na_yx_h.end()));
        if (!cond_1 and utils::is_clique(na_yx_h, A)) {
            cond_1 = true;
            for (ull sup = 0; sup < total_valid; ++sup) {
                if ((sup & sub) == sub) {
                    cond_1_list[sup] = true;
                }
            }
        }
        if (cond_1) {
            auto new_A = delete_node(x, y, H, A);
            auto pa_y = utils::pa(y, A);
            std::set<int> aux;
            std::set_union(na_yx_h.begin(), na_yx_h.end(), pa_y.begin(),
                           pa_y.end(), std::inserter(aux, aux.end()));
            aux.insert(x);
            auto old_score = cache.local_score(y, aux);
            aux.erase(x);
            auto new_score = cache.local_score(y, aux);

            if (debug) {
                std::cout << new_score - old_score << std::endl;
            }

            ++valid_count;
            if (new_score - old_score > best_score) {
                best_score = new_score - old_score;
                best_A = new_A;
                best_x = x, best_y = y;
                best_T = H;
            }
        }
    }

    return std::make_tuple(best_score, best_A, valid_count, best_x, best_y,
                           best_T);
}

auto forward_step(const torch::Tensor& A,
                  DecomposableScore& cache,
                  int debug,
                  const torch::Tensor& fixedgaps) {
    int n = A.size(0);
    int op_cnt = 0;
    torch::Tensor best_A;
    int best_x, best_y;
    std::set<int> best_T;
    double best_score = -1e10;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if ((A[i][j] == 1).item().toBool() ||
                (A[j][i] == 1).item().toBool() || i == j)
                continue;
            if (fixedgaps[i][j].item().toInt() == 1) continue;
            ++op_cnt;
            if (debug > 1)
                std::cout << "Testing operator " << i << " to " << j
                          << std::endl;
            // Get score
            auto&& [score, new_A, valid_cnt, new_x, new_y, new_T] =
                score_valid_insert_operators(i, j, A, cache,
                                             std::max(0, debug - 1));
            op_cnt += valid_cnt;
            if (score > best_score) {
                best_score = score;
                best_A = new_A;
                best_x = new_x, best_y = new_y;
                best_T = new_T;
            }
        }
    }
    if (op_cnt == 0) {
        if (debug > 1)
            std::cout << "No valid insert operators remain" << std::endl;
        return std::make_tuple(0.0, A);
    } else {
        if (debug) {
            std::cout << "Best operator: insert(" << best_x << ", " << best_y
                      << ", [";
            for (auto p : best_T) {
                std::cout << p << ",";
            }
            std::cout << "]) -> " << best_score << std::endl;
        }
        return std::make_tuple(best_score, best_A);
    }
}

auto backward_step(const torch::Tensor& A,
                   DecomposableScore& cache,
                   int debug = 0) {
    // Get candidate edges
    auto directed_fro_to = torch::where(utils::only_directed(A));
    auto fro = utils::tensor_to_int_vector(directed_fro_to[0]);
    auto to = utils::tensor_to_int_vector(directed_fro_to[1]);
    auto undirected_fro_to = torch::where(utils::only_undirected(A));
    auto un_fro = utils::tensor_to_int_vector(undirected_fro_to[0]);
    auto un_to = utils::tensor_to_int_vector(undirected_fro_to[1]);
    for (int i = 0; i < un_fro.size(); ++i) {
        if (un_fro[i] > un_to[i]) {
            fro.emplace_back(un_fro[i]);
            to.emplace_back(un_to[i]);
        }
    }

    // score
    int op_cnt = 0;
    torch::Tensor best_A;
    double best_score = -1e10;
    int best_x, best_y;
    std::set<int> best_T;

    for (int i = 0; i < fro.size(); ++i) {
        if (debug > 1) {
            std::cout << "Testing remove " << fro[i] << " to " << to[i]
                      << std::endl;
        }
        // Get score
        auto&& [score, new_A, valid_cnt, new_x, new_y, new_T] =
            score_valid_delete_operators(fro[i], to[i], A, cache,
                                         std::max(debug - 1, 0));
        op_cnt += valid_cnt;
        if (score > best_score) {
            best_A = new_A;
            best_score = score;
            best_x = new_x, best_y = new_y;
            best_T = new_T;
        }
    }

    if (op_cnt == 0) {
        if (debug > 1) {
            std::cout << "No valid delete operators remain" << std::endl;
        }
        return std::make_tuple(0.0, A);
    } else {
        if (debug) {
            std::cout << "Best operator: delete(" << best_x << ", " << best_y
                      << ", [";
            for (auto p : best_T) {
                std::cout << p << ",";
            }
            std::cout << "]) -> " << best_score << std::endl;
        }
        return std::make_tuple(best_score, best_A);
    }
}

auto fit(const torch::Tensor& A0,
         DecomposableScore& score_class,
         const std::vector<std::string>& phases = {"forward", "backward"},
         bool iterate = false,
         int debug = 0,
         const torch::Tensor& fixedgaps = at::empty({})) {
    torch::Tensor new_fixedgaps;
    if (fixedgaps.sizes().size() < 2) {
        new_fixedgaps = torch::zeros_like(A0);
    } else
        new_fixedgaps = fixedgaps;

    // GES procedure
    double total_score = 0;
    auto A = A0.clone();

    while (true) {
        auto last_total_score = total_score;
        for (const auto& phase : phases) {
            if (phase == "forward") {
                if (debug) {
                    std::cout
                        << "-----------------FORWARD----------------------"
                        << std::endl;
                }
                while (true) {
                    auto [score_change, new_A] =
                        forward_step(A, score_class, debug, new_fixedgaps);
                    if (score_change > 0.0) {
                        A = utils::pdag_to_cpdag(new_A);
                        // A = new_A.clone();
                        total_score += score_change;
                    } else
                        break;
                }
            } else if (phase == "backward") {
                if (debug) {
                    std::cout
                        << "-----------------BACKWARD----------------------"
                        << std::endl;
                }
                while (true) {
                    auto [score_change, new_A] =
                        backward_step(A, score_class, debug);
                    if (score_change > 0.0) {
                        A = utils::pdag_to_cpdag(new_A);
                        // A = new_A.clone();
                        total_score += score_change;
                    } else
                        break;
                }
            } else {
                throw "No such phase";
            }
        }
        if (total_score <= last_total_score or !iterate) {
            break;
        }
    }

    return std::make_tuple(A, total_score);
}
}  // namespace ges

#endif  // GESCPP_GES_H
