//
// Created by 谢哲 on 2021/8/8.
//

#include <iostream>
#include "ges.h"
#include <fstream>
#include <vector>
#include "torch/torch.h"
#include "DecomposableScore.h"
using namespace std;

int main() {
    ifstream inFile("/Users/xz/Project/rcit-ranker/data.pt", ios::in|ios::binary);
    vector<char> data;

    char buffer;
    while (inFile.read(&buffer, 1)) {
        data.emplace_back(buffer);
    }

    auto x = torch::pickle_load(data).toTensor();
    auto score_class = GaussObsL0Pen(x);
    auto A0 = torch::zeros({x.size(1), x.size(1)}).toType(torch::kLong);
    auto &&[result, score] = ges::fit(A0, score_class, {"forward", "backward"}, false, 1);

    return 0;
}
