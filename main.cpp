#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <set>
#include <torch/torch.h>
#include "utils.h"
#include "DecomposableScore.h"
using namespace std;
using namespace torch::indexing;

auto fit(int n,
         const torch::Tensor& A0,
         const vector<string>& phases={"forward", "backward"},
         bool iterate=false,
         int debug=0,
         const torch::Tensor& fixedgaps=at::empty({})) {
    // Check input A0 size
    assert(A0.sizes().size()==2 and n==A0.sizes()[0]);

    // GES procedure
    double total_score = 0, score_change = 1e10;
    auto A = A0;
    while(true) {
        auto last_total_score = total_score;
        double score_change = 0.0;
        for (const auto& phase: phases) {
            if (phase == "forward") {
                break;
            }
            else if (phase == "backward") {
                break;
            }
            else {
                throw exception();
            }
        }
        if (score_change <= last_total_score or !iterate) {
            break;
        }
    }
}


auto forward_step(int n, torch::Tensor& A, int debug, const torch::Tensor& fixedgaps) {
    int op_cnt = 0;
    torch::Tensor new_A;
    for (int i=0;i<n;++i) {
        for (int j=0;j<n;++j) {
            if ((A[i][j]==1).item().toBool()||(A[j][i]==1).item().toBool()||i==j)
                continue;
            ++op_cnt;
            if (debug) cout << "Testing operator " << i << " to " << j << endl;
            // Get score
        }
    }
    if (op_cnt == 0) {
        if (debug) cout << "No valid insert operators remain" << endl;
        return make_tuple(0, A);
    }
}


auto score_valid_insert_operators(
        int n,
        int x,
        int y,
        torch::Tensor A,
        int debug=0
        ) {

}


int main() {
    auto A0 = at::zeros({10, 10});
    fit(10, A0);

    return 0;
}
