#include <torch/torch.h>
#include <algorithm>
#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "DecomposableScore.h"
#include "ges.h"

int main() {
    torch::manual_seed(1234);
    auto data = torch::randn({1000, 80});
    data[500][0] = 10.0;
    data[500][9] = 20.0;
    auto A0 = torch::zeros({80, 80});

    std::cout << "Start ... " << std::endl;
    for (int i = 0; i < 1; ++i) {
        std::cout << i << std::endl;
        auto score_class = GaussObsL0Pen(data);
        auto [A, total_score] = ges::fit(A0, score_class, {"forward", "backward"}, false, 1);
    }

    std::cout << "Finished." << std::endl;
    // std::cout << A << std::endl << total_score << std::endl;

    return 0;
}
