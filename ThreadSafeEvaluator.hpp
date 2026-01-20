#pragma once
#include "ProblemData.hpp"
#include <cmath>
#include <limits>
#include <vector>


namespace LcVRPContest {

    struct EvaluationResult {
        double fitness;
        int returns;
    };

    class ThreadSafeEvaluator {
    public:
        static const int MATRIX_THRESHOLD = 5000;

        ThreadSafeEvaluator(const ProblemData& data, int num_groups);

        double Evaluate(const std::vector<int>& solution) const;
        EvaluationResult EvaluateWithStats(const std::vector<int>& solution) const;


        double GetRouteCost(const std::vector<int>& route_nodes) const;

        int GetSolutionSize() const { return num_customers_; }
        int GetLowerBound() const { return 0; }
        int GetUpperBound() const { return num_groups_ - 1; }
        const std::vector<int>& GetPermutation() const { return permutation_; }
        std::vector<int> GetDemands() const { return demands_; }

        long long GetTotalDemand() const {
            long long total = 0;
            for (int d : demands_) total += d;
            return total;
        }

        int GetDimension() const { return dimension_; }
        int GetCapacity() const { return capacity_; }
        int GetNumGroups() const { return num_groups_; }

        inline double GetDist(int i, int j) const {
            if (i == j)
                return 0.0;

            if (use_matrix_) {
                return fast_distance_matrix_[i * dimension_ + j];
            }
            else {
                double dx = coordinates_[i].x - coordinates_[j].x;
                double dy = coordinates_[i].y - coordinates_[j].y;
                return std::sqrt(dx * dx + dy * dy);
            }
        }

        bool HasMatrix() const { return use_matrix_; }
        const double* GetFastDistanceMatrix() const {
            return fast_distance_matrix_.data();
        }

        inline double GetDemand(const int c_id) const {
            if (c_id - 1 >= 0 && c_id - 1 < (int)demands_.size())
                return demands_[c_id - 1];
            return 0;
        }


        inline const Coordinate& GetCoordinate(int matrix_idx) const {
            return coordinates_[matrix_idx];
        }

        int GetTotalDepotReturns(const std::vector<int>& solution) const;
        const ProblemData& GetProblemData() const { return *problem_data_; }


        long long GetRouteCacheHits() const { return route_cache_hits_; }
        long long GetRouteCacheMisses() const { return route_cache_misses_; }


        bool HasDistanceConstraint() const { return has_distance_constraint_; }
        double GetMaxDistance() const { return max_distance_; }

    private:

        mutable long long route_cache_hits_ = 0;
        mutable long long route_cache_misses_ = 0;


        struct RouteCacheEntry {
            uint64_t key = 0;
            double cost = -1.0;
            int returns = -1;
            bool occupied = false;
        };


        static constexpr size_t CACHE_SIZE = 1 << 25;
        static constexpr size_t CACHE_MASK = CACHE_SIZE - 1;

        mutable std::vector<RouteCacheEntry> route_cache_;
        std::vector<uint64_t> customer_hashes_;

        void InitCache();


        inline uint64_t GetCustomerHash(int customer_id) const {
            return customer_hashes_[customer_id];
        }


        int num_groups_;
        int num_customers_;
        int dimension_;
        int capacity_;
        int depot_index_;
        bool has_distance_constraint_;
        double max_distance_;
        const ProblemData* problem_data_;

        bool use_matrix_;

        std::vector<int> demands_;
        std::vector<int> permutation_;

        std::vector<double> fast_distance_matrix_;
        std::vector<Coordinate> coordinates_;
    };
}