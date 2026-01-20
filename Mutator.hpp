#pragma once
#include "Individual.hpp"
#include "ProblemGeometry.hpp"
#include "Split.hpp"
#include "ThreadSafeEvaluator.hpp"
#include "AdaptiveOperators.hpp"
#include <algorithm>
#include <random>
#include <vector>

namespace LcVRPContest {

    class Mutator {
    public:
        Mutator();


        void Initialize(ThreadSafeEvaluator* eval, const ProblemGeometry* geo,
            Split* split);




        bool ApplySmartSpatialMove(Individual& indiv, std::mt19937& rng);


        bool AggressiveMutate(Individual& indiv, std::mt19937& rng);




        bool ApplyMicroSplitMutation(Individual& indiv, double stagnation_factor,
            int level, std::mt19937& rng);


        bool ApplySimpleMutation(Individual& indiv, std::mt19937& rng);




        bool ApplyRuinRecreate(Individual& indiv, double intensity,
            bool is_exploitation, std::mt19937& rng);

        bool ApplyAdaptiveLNS(Individual& indiv, LNSStrategy strategy,
            double intensity, bool is_exploitation, std::mt19937& rng);



        bool ApplyReturnMinimizer(Individual& indiv, std::mt19937& rng);


        bool ApplyMergeSplit(Individual& indiv, std::mt19937& rng);

        bool EliminateReturns(Individual& indiv, std::mt19937& rng);

    private:

        std::vector<int> merge_route_buffer_;
        ThreadSafeEvaluator* evaluator_;
        const ProblemGeometry* geometry_;
        Split* split_ptr_;


        std::vector<int> removed_indices_buffer_;
        std::vector<bool> is_removed_buffer_;


        std::vector<int> candidates_buffer_;


        std::vector<int> group_votes_buffer_;


        std::vector<int> group_loads_buffer_;
        std::vector<std::vector<int>> group_clients_buffer_;
        std::vector<std::pair<int, int>> group_overload_buffer_;
        std::vector<std::pair<double, int>> relatedness_buffer_;


        std::vector<std::vector<int>> routes_buffer_;
        std::vector<int> overflow_groups_buffer_;
        std::vector<int> check_indices_buffer_;
        std::vector<int> temp_route_buffer_;


        int GetClientId(int idx) const;
        std::vector<int> EncodeRoute(const std::vector<int>& idx_route) const;


        std::vector<int> customer_ranks_;
    };

}
