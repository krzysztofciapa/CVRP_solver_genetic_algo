#include "ProblemGeometry.hpp"
#include <algorithm>
#include <chrono>

using namespace LcVRPContest;

ProblemGeometry::ProblemGeometry(const ProblemData& data, int id) : id_(id) {
    rng_.seed(static_cast<unsigned int>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() +
        id * 999));
}

void ProblemGeometry::Initialize(ThreadSafeEvaluator* evaluator) {
    PrecomputeNeighbors(evaluator);
}

void ProblemGeometry::PrecomputeNeighbors(ThreadSafeEvaluator* evaluator) {
    int n = evaluator->GetSolutionSize();
    neighbors_.assign(n, std::vector<int>());

    const auto& perm = evaluator->GetPermutation();
    int dim = evaluator->GetDimension();

    std::vector<int> client_rank(dim + 1, -1);

    for (size_t r = 0; r < perm.size(); ++r) {
        if (perm[r] > 1) {
            client_rank[perm[r]] = static_cast<int>(r);
        }
    }

    for (int i = 0; i < n; ++i) {
        int u_id = i + 2;
        if (u_id > dim) continue;

        int u_node_idx = u_id - 1;

        std::vector<int> logical_neighbors;
        int rank = client_rank[u_id];

        if (rank != -1) {

            if (rank > 0) {
                int prev_id = perm[rank - 1];
                if (prev_id > 1) {

                    logical_neighbors.push_back(prev_id - 2);
                }
            }
            if (rank < (int)perm.size() - 1) {
                int next_id = perm[rank + 1];
                if (next_id > 1) {
                    logical_neighbors.push_back(next_id - 2);
                }
            }
        }

        std::vector<std::pair<double, int>> dists;
        dists.reserve(n);

        for (int j = 0; j < n; ++j) {
            if (i == j) continue;

            int v_id = j + 2;
            int v_node_idx = v_id - 1;

            double d = evaluator->GetDist(u_node_idx, v_node_idx);
            dists.push_back({ d, j });
        }

        int neighbor_limit = Config::NUM_NEIGHBORS;
        if (n > Config::HUGE_INSTANCE_THRESHOLD) {
            neighbor_limit = 12;
        }

        size_t keep_geo = std::min((size_t)(neighbor_limit + 4), dists.size());

        std::nth_element(dists.begin(), dists.begin() + keep_geo, dists.end());

        std::sort(dists.begin(), dists.begin() + keep_geo);


        neighbors_[i].reserve(neighbor_limit + 4);

        for (int logical : logical_neighbors) {
            neighbors_[i].push_back(logical);
        }

        for (size_t k = 0; k < keep_geo; ++k) {
            int geo_idx = dists[k].second;

            bool is_duplicate = false;
            for (int existing : neighbors_[i]) {
                if (existing == geo_idx) {
                    is_duplicate = true;
                    break;
                }
            }

            if (!is_duplicate) {
                neighbors_[i].push_back(geo_idx);
            }
            if (neighbors_[i].size() >= (size_t)neighbor_limit + 2) break;
        }
    }
}
