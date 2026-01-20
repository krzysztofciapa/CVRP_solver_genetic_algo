#pragma once
#include "Individual.hpp"
#include "Split.hpp"
#include "ThreadSafeEvaluator.hpp"

#include <cstdint>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace LcVRPContest {


    struct CachedRoute {
        std::vector<int> nodes;
        double cost = 0.0;
        double efficiency = 0.0;
        uint64_t hash = 0;
        std::vector<uint64_t> bitmask;

        bool operator<(const CachedRoute& other) const {
            return efficiency < other.efficiency;
        }
    };


    class RoutePool {
    public:

        void AddRoutesFromSolution(const std::vector<int>& solution, const ThreadSafeEvaluator& evaluator);


        Individual SolveBeamSearch(ThreadSafeEvaluator* evaluator, Split& split, int beam_width);


        void ImportRoutes(const std::vector<CachedRoute>& imported_routes);


        std::vector<CachedRoute> GetBestRoutes(int n) const;


        void Clear();
        size_t GetSize() const;


        bool HasNewRoutesSince(size_t snapshot) const {
            return total_routes_added_ > snapshot;
        }


        size_t GetTotalRoutesAdded() const {
            return total_routes_added_;
        }

    private:

        double CalculateRouteCost(const std::vector<int>& route, const ThreadSafeEvaluator& evaluator) const;

        uint64_t HashRoute(const std::vector<int>& sorted_route) const;

        void EvictWorstRoutes();

        std::vector<CachedRoute> routes_;
        std::unordered_set<uint64_t> route_hashes_;
        mutable std::mutex mutex_;
        size_t total_routes_added_ = 0;
    };

}
