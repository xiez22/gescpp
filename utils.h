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
        auto new_A = A;
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
}

#endif //GESCPP_UTILS_H
