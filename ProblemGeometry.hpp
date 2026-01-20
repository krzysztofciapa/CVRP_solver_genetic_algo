#pragma once
#include "Constants.hpp"
#include "ProblemData.hpp"
#include "ThreadSafeEvaluator.hpp"
#include <random>
#include <vector>


namespace LcVRPContest {

    class ProblemGeometry {
    public:
        ProblemGeometry(const ProblemData& data, int id);

        void Initialize(ThreadSafeEvaluator* evaluator);

        inline const std::vector<int>& GetNeighbors(int index) const {
            static const std::vector<int> empty;
            if (index < 0 || index >= (int)neighbors_.size())
                return empty;
            return neighbors_[index];
        }

        bool HasNeighbors() const { return !neighbors_.empty(); }

    private:
        int id_;
        std::mt19937 rng_;

        std::vector<std::vector<int>> neighbors_;

        void PrecomputeNeighbors(ThreadSafeEvaluator* evaluator);
    };
}
