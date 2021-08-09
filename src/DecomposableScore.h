//
// Created by 谢哲 on 2021/8/5.
//

#ifndef GESCPP_DECOMPOSABLESCORE_H
#define GESCPP_DECOMPOSABLESCORE_H
#include <cmath>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>
#include "torch/torch.h"

class DecomposableScore {
   protected:
    bool cache = true;
    int debug = 0;
    std::map<std::vector<int>, double> _cache;

   public:
    explicit DecomposableScore(bool cache = true, int debug = 0)
        : cache(cache), debug(debug) {}

    double local_score(int x, const std::set<int>& pa) {
        if (debug) {
            std::cout << x << "(";
            for (auto p : pa)
                std::cout << "," << p;
            std::cout << ") :";
        }
        double value;
        if (!cache) {
            value = _compute_local_score(x, pa);
        } else {
            std::vector key = {x};
            key.insert(key.end(), pa.begin(), pa.end());
            if (_cache.find(key) != _cache.end()) {
                if (debug) std::cout << "using cached value ";
                value = _cache[key];
            } else {
                value = _cache[key] = _compute_local_score(x, pa);
            }
        }
        return value;
    }

    [[nodiscard]] virtual double _compute_local_score(
        int x,
        const std::set<int>& pa) const {
        return 0.0;
    }
};

class GaussObsL0Pen : public DecomposableScore {
   public:
    torch::Tensor data, _centered;
    int n, p;
    double lmbda;

    explicit GaussObsL0Pen(torch::Tensor _data,
                           bool cache = true,
                           int debug = 0)
        : data(std::move(_data)), DecomposableScore(cache, debug) {
        n = data.size(0);
        lmbda = 0.5 * log(n);
        p = data.size(1);
        _centered = data - data.mean(0);
    }

    [[nodiscard]] double _compute_local_score(int x, const std::set<int>& pa)
        const override {
        torch::Tensor sigma;
        sigma = _mle_local(x, pa);
        auto likelihood = -0.5 * n * (1.0 + torch::log(sigma));
        auto l0_term = lmbda * double(pa.size() + 1);
        auto score = likelihood.item().toDouble() - l0_term;
        return score;
    }

    [[nodiscard]] torch::Tensor _mle_local(int j,
                                           const std::set<int>& parents) const {
        std::vector<int> parents_vec{parents.begin(), parents.end()};
        auto parents_torch = torch::tensor(parents_vec);
        auto Y = _centered.index({"...", j});
        torch::Tensor sigma;
        if (!parents.empty()) {
            auto X = torch::atleast_2d(_centered.index({"...", parents_torch}));
            auto [coef, u1, u2, u3] =
                torch::linalg::lstsq(X, Y, c10::nullopt, c10::nullopt);
            sigma = torch::var(Y - torch::matmul(X, coef));
        } else {
            sigma = torch::var(Y);
        }
        return sigma;
    }
};

class GaussClusterL0Pen : public DecomposableScore {
   public:
    torch::Tensor data, _centered;
    int n, p;
    double lmbda;
    std::vector<std::vector<int>> graph;

    explicit GaussClusterL0Pen(torch::Tensor _data,
                               std::vector<std::vector<int>> graph,
                               bool cache = true,
                               int debug = 0)
        : data(std::move(_data)),
          DecomposableScore(cache, debug),
          graph(std::move(graph)) {
        n = (int)data.size(0);
        lmbda = 0.5 * log(n);
        p = (int)data.size(1);
        _centered = data - data.mean(0);
    }

    [[nodiscard]] double _compute_local_score(int x, const std::set<int>& pa)
        const override {
        auto x_single = graph[x];
        std::set<int> pa_single;
        for (auto i : pa) {
            for (auto j : graph[i]) {
                pa_single.insert(j);
            }
        }

        double max_score = -1e10;
        for (auto i : x_single) {
            auto result = _compute_single_local_score(i, pa_single);
            max_score = std::max(max_score, result);
        }

        return max_score;
    }

    [[nodiscard]] double _compute_single_local_score(
        int x,
        const std::set<int>& pa) const {
        torch::Tensor sigma;
        sigma = _mle_local(x, pa);
        auto likelihood = -0.5 * n * (1.0 + torch::log(sigma));
        auto l0_term = lmbda * double(pa.size() + 1);
        auto score = likelihood.item().toDouble() - l0_term;
        return score;
    }

    [[nodiscard]] torch::Tensor _mle_local(int j,
                                           const std::set<int>& parents) const {
        std::vector<int> parents_vec{parents.begin(), parents.end()};
        auto parents_torch = torch::tensor(parents_vec);
        auto Y = _centered.index({"...", j});
        torch::Tensor sigma;
        if (!parents.empty()) {
            auto X = torch::atleast_2d(_centered.index({"...", parents_torch}));
            auto [coef, u1, u2, u3] =
                torch::linalg::lstsq(X, Y, c10::nullopt, c10::nullopt);
            sigma = torch::var(Y - torch::matmul(X, coef));
        } else {
            sigma = torch::var(Y);
        }
        return sigma;
    }
};

#endif  // GESCPP_DECOMPOSABLESCORE_H
