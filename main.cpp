#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <algorithm>
#include <set>
#include <torch/torch.h>
#include "utils.h"
#include "DecomposableScore.h"
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

auto score_valid_insert_operators(
        int x,
        int y,
        const torch::Tensor& A,
        DecomposableScore& cache,
        int debug=0) {
    int p = A.size(0);
    auto s1 = utils::neighbors(y, A), s2 = utils::adj(x, A);
    std::vector<int> T0;
    std::set_difference(s1.begin(), s1.end(), s2.begin(), s2.end(), std::back_inserter(T0));

    ull total_valid = (1<<T0.size());
    std::vector<bool> removed(total_valid, false);
    std::vector<bool> passed_cond_2(total_valid, false);

    int valid_count = 0;
    double best_score = -1e10;
    torch::Tensor best_A;

    // Traverse all subsets of T0
    for (ull sub=0;sub<total_valid;++sub) {
        if (removed[sub]) continue;
        // Check Cond 1
        std::set<int> T;
        for (int i=0;i<T0.size();++i)
            if (((1ull<<i)&sub)==(1ull<<i)) T.insert(i);

        auto yxT = utils::na(y, x, A);
        std::set<int> na_yxT;
        set_union(yxT.begin(), yxT.end(), T.begin(), T.end(), std::inserter(na_yxT, na_yxT.end()));
        auto cond_1 = utils::is_clique(na_yxT, A);

        if (!cond_1) {
            // Remove from other which contain T
            ull T_bits = 0;
            for (auto node: T) T_bits |= 1ull << node;
            for (ull sup=0;sup<total_valid;++sup)
                if ((sup&T_bits)==T_bits) removed[sup] = true;
        }
        bool cond_2;
        if (passed_cond_2[sub]) {
            cond_2 = true;
        }
        else {
            cond_2 = true;
            for (const auto& path: utils::semi_directed_paths(y, x, A)) {
                int cnt = 0;
                for (auto node: path) {
                    if (na_yxT.contains(node)) ++cnt;
                }
                if (cnt == 0) {
                    cond_2 = false;
                    break;
                }
            }
            if (cond_2) {
                ull T_bits = 0;
                for (auto node: T) T_bits |= 1ull << node;
                for (ull sup=0;sup<total_valid;++sup)
                    if ((sup&T_bits)==T_bits) passed_cond_2[sup] = true;
            }
        }
        if (cond_1 and cond_2) {
            auto new_A = insert(x, y, T, A);
            std::set<int> aux;
            auto pa_y = utils::pa(y, A);
            set_union(na_yxT.begin(), na_yxT.end(), pa_y.begin(), pa_y.end(), std::inserter(aux, aux.end()));
            // Compute the change in score
            auto old_score = cache.local_score(y, aux);
            aux.insert(x);
            auto new_score = cache.local_score(y, aux);
            if (isinf(old_score) or isinf(new_score)) continue;
            if (debug) std::cout<<new_score-old_score<<std::endl;

            valid_count++;
            if (new_score-old_score > best_score) {
                best_score = new_score - old_score;
                best_A = new_A;
            }
        }
    }

    return std::make_tuple(best_score, best_A, valid_count);
}

auto forward_step(torch::Tensor& A,
                  DecomposableScore& cache,
                  int debug,
                  const torch::Tensor& fixedgaps) {
    int n = A.size(0);
    int op_cnt = 0;
    torch::Tensor best_A;
    double best_score = -1e10;

    for (int i=0;i<n;++i) {
        for (int j=0;j<n;++j) {
            if ((A[i][j]==1).item().toBool()||(A[j][i]==1).item().toBool()||i==j)
                continue;
            if (fixedgaps[i][j].item().toInt()==1) continue;
            ++op_cnt;
            if (debug) std::cout << "Testing operator " << i << " to " << j << std::endl;
            // Get score
            auto &&[score, new_A, valid_cnt] = score_valid_insert_operators(i, j, A, cache, debug);
            op_cnt += valid_cnt;
            if (score > best_score) {
                best_score = score;
                best_A = new_A;
            }
        }
    }
    if (op_cnt == 0) {
        if (debug) std::cout << "No valid insert operators remain" << std::endl;
        return std::make_tuple(0.0, A);
    }
    else {
        if (debug) std::cout << "----------------------> Best score: " << best_score << std::endl;
        return std::make_tuple(best_score, best_A);
    }
}

auto fit(int n,
         const torch::Tensor& A0,
         DecomposableScore& score_class,
         const std::vector<std::string>& phases={"forward", "backward"},
         bool iterate=false,
         int debug=0,
         const torch::Tensor& fixedgaps=at::empty({})) {
    // Check input A0 size
    assert(A0.sizes().size()==2 and n==A0.sizes()[0]);

    torch::Tensor new_fixedgaps;
    if (fixedgaps.sizes().size() < 2) {
        new_fixedgaps = torch::zeros_like(A0);
    }
    else new_fixedgaps = fixedgaps;

    // GES procedure
    double total_score = 0;
    auto A = A0.clone();

    while(true) {
        auto last_total_score = total_score;
        for (const auto& phase: phases) {
            if (phase == "forward") {
                while(true) {
                    auto [score_change, new_A] = forward_step(A, score_class, debug, new_fixedgaps);
                    if (score_change > 0.0) {
                        //A = utils::pdag_to_cpdag(new_A);
                        A = new_A.clone();
                        total_score += score_change;
                    }
                    else break;
                }
            }
            else if (phase == "backward") {
                break;
            }
            else {
                throw "No such phase";
            }
        }
        if (total_score <= last_total_score or !iterate) {
            break;
        }
    }

    return std::make_tuple(A, total_score);
}

int main() {
    // auto A0 = at::zeros({10, 10});
    // fit(10, A0);
    torch::manual_seed(1234);
    auto data = torch::randn({1000, 50});
    data[500][0] = 10.0;
    data[500][9] = 20.0;
    auto A0 = torch::zeros({50, 50});

    std::cout << "Start ... " << std::endl;
    for (int i=0;i<1;++i) {
        std::cout << i << std::endl;
        auto score_class = GaussObsL0Pen(data);
        auto [A, total_score] = fit(50, A0, score_class, {"forward"}, false, 0);
    }

    std::cout << "Finished." << std::endl;
    //std::cout << A << std::endl << total_score << std::endl;

    return 0;
}
