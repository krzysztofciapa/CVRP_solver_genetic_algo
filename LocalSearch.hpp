#pragma once
#include "Constants.hpp"
#include "Individual.hpp"
#include "ProblemGeometry.hpp"
#include "ThreadSafeEvaluator.hpp"
#include <random>
#include <vector>

namespace LcVRPContest {


    struct RoutePosition {
        int prev_client;
        int next_client;
        int route_id;
        int position;
    };


    struct SectorInfo {
        std::vector<int> client_indices;
        int seed_client;
    };

    class LocalSearch {
    public:
        LocalSearch(ThreadSafeEvaluator* evaluator, const ProblemGeometry* geometry,
            int id);


        bool RunVND(Individual& ind, bool heavy_mode = false);
        bool RunVND(Individual& ind, int max_iter, bool allow_swap,
            bool allow_3swap = false, bool allow_ejection = false,
            bool allow_4swap = false, bool unlimited_moves = false);

        bool RunHugeInstanceVND(Individual& ind, int tier);




        bool Try3Swap(std::vector<int>& genotype);

        bool Try4Swap(std::vector<int>& genotype);


        void SetGuideSolution(const std::vector<int>& guide) { guide_solution_ = guide; }
        const std::vector<int>& GetGuideSolution() const { return guide_solution_; }



        bool TryPathRelinking(std::vector<int>& genotype, double& current_cost,
            const std::vector<int>& guide_solution);


        bool TryEjectionChain(std::vector<int>& genotype, int start_client_idx,
            int max_depth = 3);

        long long prsucc = 0;
    private:
        ThreadSafeEvaluator* evaluator_;
        const ProblemGeometry* geometry_;
        int id_;
        std::mt19937 rng_;


        const double* fast_matrix_ = nullptr;
        int matrix_dim_ = 0;


        std::vector<std::vector<int>> vnd_routes_;
        std::vector<double> vnd_loads_;
        std::vector<double> route_costs_;
        std::vector<int> max_cumulative_load_;

        std::vector<int> temp_route_buffer_;

        std::vector<int> customer_ranks_;
        std::vector<int> client_indices_;
        std::vector<int> candidate_groups_;


        std::vector<RoutePosition> positions_;

        void InitializeRanks();
        void BuildPositions();
        void
            BuildCumulativeLoads();
        void UpdatePositionsAfterMove(int client_id, int old_route, int new_route);




        bool IsSafeMove(int target_route, int client_id) const;

        double CalculateFastInsertionDelta(int client_id, int target_route,
            int insert_pos) const;

        double SimulateRouteCostWithInsert(int target_route, int client_id,
            int insert_pos) const;
        double SimulateRouteCostWithRemoval(int source_route, int client_id) const;


        bool RunFullVND(Individual& ind, bool allow_swap);

        bool OptimizeActiveSet(Individual& ind, int max_iter, bool allow_swap,
            bool allow_3swap, bool allow_ejection = false,
            bool allow_4swap = false, bool unlimited_moves = false);


        std::vector<int> guide_solution_;

        std::vector<bool> dlb_;

        double CalculateRemovalDelta(int client_id) const;

        double CalculateInsertionDelta(int client_id, int target_route, int& best_insert_pos) const;


        double SimulateRouteCost(const std::vector<int>& route_nodes) const;


        bool WouldOverflow(int target_route, int client_id) const;

        void ResetDLB();

        std::vector<SectorInfo> sectors_;
        std::vector<int> client_sector_;
        bool sectors_initialized_ = false;
    };
}
