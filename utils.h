//
// Created by 谢哲 on 2021/8/5.
//

#ifndef GESCPP_UTILS_H
#define GESCPP_UTILS_H
#include <set>
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include "torch/torch.h"
using namespace torch::indexing;

namespace utils {
    auto neighbors(int i, const torch::Tensor& A) {
        std::set<int> result;
        int n = A.size(0);

        for (int j=0;j<n;++j) {
            if (A[i][j].item().toInt()!=0 and A[j][i].item().toInt()!=0) {
                result.insert(j);
            }
        }
        return result;
    }

    auto adj(int i, const torch::Tensor& A) {
        std::set<int> result;
        int n = A.size(0);

        for (int j=0;j<n;++j) {
            if (A[i][j].item().toInt()!=0 or A[j][i].item().toInt()!=0) {
                result.insert(j);
            }
        }
        return result;
    }

    auto na(int y, int x, const torch::Tensor& A) {
        auto s1 = neighbors(y, A);
        auto s2 = adj(x, A);
        std::set<int> result;
        std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(), std::inserter(result, result.end()));
        return result;
    }

    auto pa(int i, const torch::Tensor& A) {
        std::set<int> result;
        int n = A.size(0);

        for (int j=0;j<n;++j) {
            if (A[j][i].item().toInt()!=0 and A[i][j].item().toInt()==0) {
                result.insert(j);
            }
        }
        return result;
    }

    auto ch(int i, const torch::Tensor& A) {
        std::set<int> result;
        int n = A.size(0);

        for (int j=0;j<n;++j) {
            if (A[i][j].item().toInt()!=0 and A[j][i].item().toInt()==0) {
                result.insert(j);
            }
        }
        return result;
    }

    auto skeleton(const torch::Tensor& A) {
        auto result = ((A + A.t()) != 0).toType(torch::kI64);
        return result;
    }

    auto is_clique(const std::set<int>& S, const torch::Tensor& A) {
        std::vector<int> S_vec{S.begin(), S.end()};
        auto S_torch = torch::tensor(S_vec);
        auto subgraph = A.index({S_torch, "..."}).index({"...", S_torch});
        int no_edges = torch::sum(skeleton(subgraph)!=0).item().toInt();
        int n = S.size();
        return no_edges == n * (n-1);
    }

    auto only_directed(const torch::Tensor& P) {
        auto mask = torch::logical_and(P!=0, P.t()==0);
        auto G = torch::zeros_like(P);
        G.index_put_({mask}, P.index({mask}));
        return G;
    }

    auto only_undirected(const torch::Tensor& P) {
        auto mask = torch::logical_and(P!=0, P.t()!=0);
        auto G = torch::zeros_like(P);
        G.index_put_({mask}, P.index({mask}));
        return G;
    }

    auto topological_ordering(const torch::Tensor& A) {
        if (only_undirected(A).sum().item().toInt() > 0) {
            throw "The given graph is not a DAG";
        }
        auto new_A = A.clone();
        auto sinks_torch = torch::where(new_A.sum({0})==0)[0].contiguous();
        std::vector<int> sinks{sinks_torch.data_ptr<int>(), sinks_torch.data_ptr<int>()+sinks_torch.numel()};
        std::vector<int> ordering;

        while(!sinks.empty()) {
            auto i = sinks.back();
            sinks.pop_back();
            ordering.emplace_back(i);
            for (auto j: ch(i, new_A)) {
                new_A[i][j] = 0;
                if (pa(j, new_A).empty()) {
                    sinks.emplace_back(j);
                }
            }
        }

        if (new_A.sum().item().toInt() > 0) {
            throw "The given graph is not a DAG";
        }
        else {
            return ordering;
        }
    }

    auto is_dag(const torch::Tensor& A) {
        try {
            topological_ordering(A);
            return true;
        }
        catch (...) {
            return false;
        }
    }

    auto semi_directed_paths(int fro, int to, const torch::Tensor& A) {
        // Initialize map
        std::unordered_map<int, std::vector<int>> mdata;
        int n = A.size(0);
        for (int i=0;i<n;++i)
            for (int j=0;j<n;++j)
                if (A[i][j].item().toInt()!=0)
                    mdata[i].emplace_back(j);

        // Dfs
        std::vector<bool> visited(n, false);
        std::vector<std::vector<int>> paths;
        std::vector<int> current_path;
        std::function<void(int)> dfs = [&](int pos) {
            current_path.emplace_back(pos);
            visited[pos] = true;
            if (pos == to) {
                paths.emplace_back(current_path);
            }
            else {
                for (auto p: mdata[pos]) {
                    if (visited[p]) continue;
                    dfs(p);
                }
            }
            current_path.pop_back();
            visited[pos] = false;
        };
        dfs(fro);
        return paths;
    }

    auto pdag_to_dag(const torch::Tensor& _P) {
        auto P = _P.clone();
        auto G = only_directed(P);
        std::vector<int> indexes(P.size(0));
        for (int i=0;i<indexes.size();++i) indexes[i] = i;

        while (P.size(0) > 0) {
            auto found = false;
            int i = 0;
            while (!found and i < P.size(0)) {
                // Check condition 1
                auto sink = (ch(i, P).size() == 0);
                // Check condition 2
                auto n_i = neighbors(i, P);
                auto adj_i = adj(i, P);
                bool adj_neighbors = true;
                for (auto y: n_i) {
                    auto adj_yP = adj(y, P);
                    for (auto node: adj_i) {
                        if (node == y) continue;
                        if (!adj_yP.contains(node)) {
                            adj_neighbors = false;
                            break;
                        }
                    }
                    if (!adj_neighbors) break;
                }
                found = sink and adj_neighbors;
                // Orient all incident undirected edges and remove i
                if (found) {
                    auto real_i = indexes[i];
                    std::vector<int> real_neighbors(n_i.size());
                    real_neighbors.reserve(n_i.size());
                    for (auto j: n_i) real_neighbors.emplace_back(indexes[j]);
                    for (auto j: real_neighbors) G.index_put_({j, real_i}, 1);
                    std::vector<int> all_but_i;
                    for (int j=0;j<P.size(0)-1;++j) if (i!=j) all_but_i.emplace_back(j);
                    auto all_but_i_torch = torch::tensor(all_but_i);
                    P = P.index({all_but_i_torch, "..."}).index({"...", all_but_i_torch});
                    indexes.erase(std::find(indexes.begin(), indexes.end(), real_i));
                }
                else ++i;
            }

            if (!found) {
                throw "PDAG does not admit consistent extension";
            }
        }

        return G;
    }

    auto order_edges(const torch::Tensor& G) {
        auto order = topological_ordering(G);
        auto ordered = (G!=0).toType(torch::kLong) * -1;
        int i = 1;
        while (torch::any(ordered).item().toBool()) {
            auto ul = torch::hstack(torch::where(ordered==-1)).contiguous();
            std::set<int> with_unlablled{ul.data_ptr<int>(), ul.data_ptr<int>()+ul.numel()};
            int y = 0;
            for (auto p=order.rbegin(); p!=order.rend(); ++p) {
                if (with_unlablled.contains(*p)) {
                    y = *p;
                    break;
                }
            }
            auto ul_y = torch::where(ordered.index({"...", y})==-1)[0].contiguous();
            std::set<int> unlabelled_parents_y{ul_y.data_ptr<int>(), ul_y.data_ptr<int>()+ul.numel()};
            int x = 0;
            for (auto p: order) {
                if (unlabelled_parents_y.contains(p)) {
                    x = p;
                    break;
                }
            }
            ordered[x][y] = i;
            ++i;
        }

        return ordered;
    }

    auto label_edges(const torch::Tensor& ordered) {
        // define labels: 1: compelled, -1: reversible, -2: unknown
        int COM = 1, REV = -1, UNK = -2;
        auto labelled = (ordered!=0).toType(torch::kLong) * UNK;

        while (torch::any(labelled==UNK).item().toBool()) {
            auto unknown_edges = (ordered * (labelled==UNK).toType(torch::kLong)).toType(torch::kFloat);
            unknown_edges.index_put_({unknown_edges==0}, -1e10);
            auto max_pos = torch::argmax(unknown_edges).item().toInt();
            int x = max_pos / (int)unknown_edges.size(0), y = max_pos % (int)unknown_edges.size(0);
            auto Ws = torch::where(labelled.index({"...", x})==COM)[0];
            auto end = false;

            int Ws_len = (int)Ws.size(0);
            for (int w=0;w<Ws_len;++w) {
                if (labelled[w][y].item().toInt() == 0) {
                    auto pa_y = pa(y, labelled);
                    auto pa_y_torch = torch::tensor(std::vector<int>{pa_y.begin(), pa_y.end()});
                    end = true;
                    break;
                }
                else {
                    labelled[w][y] = COM;
                }
            }

            if (!end) {
                auto s1 = pa(y, labelled), s2 = pa(x, labelled);
                s2.insert(x);
                std::vector<int> result;
                std::set_difference(s1.begin(), s1.end(), s2.begin(), s2.end(), std::back_inserter(result));
                auto z_exists = result.size() > 0;
                auto unknown = torch::where(labelled.index({"...", y})==UNK)[0];
                if (z_exists) {
                    labelled.index_put_({unknown, y}, COM);
                }
                else {
                    labelled.index_put_({unknown, y}, REV);
                }
            }
        }

        return labelled;
    }

    auto dag_to_cpdag(const torch::Tensor& G) {
        auto ordered = order_edges(G);
        auto labelled = label_edges(ordered);
        auto cpdag = torch::zeros_like(labelled);
        cpdag.index_put_({labelled==1}, labelled.index({labelled==1}));
        int l1 = (int)labelled.size(0), l2 = (int)labelled.size(1);
        for (int x=0;x<l1;++x) {
            for (int y=0;y<l2;++y) {
                if (labelled[x][y].item().toInt()==-1) {
                    cpdag[x][y] = 1;
                    cpdag[y][x] = 1;
                }
            }
        }
        return cpdag;
    }
}

#endif //GESCPP_UTILS_H
