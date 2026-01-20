#pragma once
#include "Individual.hpp"
#include "ProblemGeometry.hpp"
#include "ThreadSafeEvaluator.hpp"
#include <deque>
#include <vector>

#include <random>

namespace LcVRPContest {

    struct SplitResult {
        bool feasible;
        double total_cost;
        std::vector<std::vector<int>> optimized_routes;
        std::vector<int> group_assignment;
    };

    struct Segment {
        int start_k;
        int end_k;
        double demand;
    };

    class Split {
    public:
        Split(const ThreadSafeEvaluator* evaluator);

        SplitResult RunLinear(const std::vector<int>& giant_tour);
        SplitResult RunBellman(const std::vector<int>& giant_tour);

        void ApplyMicroSplit(Individual& indiv, int start_idx, int end_idx,
            const ProblemGeometry* geometry, std::mt19937& rng);

    private:
        const ThreadSafeEvaluator* evaluator_;
        int capacity_;
        int depot_idx_;
        int num_customers_;

        std::vector<double> D_;
        std::vector<int> Q_;
        std::vector<double> V_;
        std::vector<int> pred_;
        std::deque<int> dq_;

        std::vector<int> votes_buffer_;
        std::vector<Segment> segments_buffer_;
        std::vector<bool> in_window_;

        void ResizeStructures(int size);
        void PrecomputeStructures(const std::vector<int>& giant_tour);
        SplitResult ReconstructResult(const std::vector<int>& giant_tour);
    };

}
