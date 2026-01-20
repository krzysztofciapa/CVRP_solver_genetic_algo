#pragma once
#include "Constants.hpp"
#include "Evaluator.hpp"
#include "Individual.hpp"
#include "LocalCache.hpp"
#include "ProblemData.hpp"
#include "Split.hpp"
#include "ThreadSafeEvaluator.hpp"


#include "LocalSearch.hpp"
#include "Mutator.hpp"
#include "ProblemGeometry.hpp"
#include "RoutePool.hpp"

#include <array>
#include <atomic>
#include <deque>
#include <mutex>
#include <random>
#include <vector>

namespace LcVRPContest {

    enum class INITIALIZATION_TYPE {
        RANDOM,
        CHUNKED,
        RR,
        SMART_STICKY,
        BIN_PACKING,
        K_CENTER_CLUSTERING
    };

    class Island {
    public:
        Island(ThreadSafeEvaluator* evaluator, const ProblemData& data,
            int population_size, int id);


        ~Island() = default;
        Island(const Island&) = delete;
        Island& operator=(const Island&) = delete;
        Island(Island&&) = default;
        Island& operator=(Island&&) = default;

        void Initialize(INITIALIZATION_TYPE strategy);
        void RunGeneration();
        void InjectImmigrant(Individual& imigrant, bool force = false);
        void SetRingPredecessor(Island* pred) { ring_predecessor_ = pred; }
        void TryPullMigrant();



        void SetExploitSiblings(std::vector<Island*> siblings) {
            exploit_siblings_ = siblings;
        }
        void ReceiveBroadcastBest(
            const Individual& best);

        double EvaluateWithHistoryPenalty(const std::vector<int>& genotype);

        std::vector<int> GetBestSolution() const {
            std::lock_guard<std::mutex> lock(best_mutex_);
            return current_best_.GetGenotype();
        }

        Individual GetBestIndividual() const {
            std::lock_guard<std::mutex> lock(best_mutex_);
            return current_best_;
        }




        Individual GetRandomEliteIndividual();


        std::vector<CachedRoute> GetTopRoutes(int n) const;

        double GetBestFitness() const {
            std::lock_guard<std::mutex> lock(best_mutex_);
            return current_best_.GetFitness();
        }



        int GetId() const { return id_; }
        bool IsExploration() const { return (id_ % 2) == 0; }
        bool IsExploitation() const { return (id_ % 2) == 1; }


        bool ContainsSolution(const Individual& ind) const;
        double GetCurrentStructuralDiversity() const { return current_structural_diversity_; }


        long long GetCacheHits() const { return cache_hits_; }
        long long GetCacheMisses() const { return cache_misses_; }
        double GetCacheMissRate() const;


        double GetPopulationHealth() const;


        double GetRecentCacheHitRate() const;






        void SetStartTime(std::chrono::steady_clock::time_point start_time) {
            start_time_ = start_time;
        }




        int CalculateBrokenPairsDistancePublic(const Individual& ind1,
            const Individual& ind2) {
            return CalculateBrokenPairsDistance(
                ind1, ind2, evaluator_->GetPermutation(), evaluator_->GetNumGroups());
        }



        long long getPRStats() const { return local_search_.prsucc; }

    private:
        ThreadSafeEvaluator* evaluator_;

        Mutator mutator_;
        AdaptiveOperatorSelector aos_;
        const std::vector<int>& demands_;
        int capacity_;

        ProblemGeometry geometry_;
        LocalSearch local_search_;

        std::vector<int> customer_ranks_;
        int population_size_;
        int id_;

        LocalCache local_cache_;
        std::vector<Individual> population_;

        Individual current_best_;
        std::mt19937 rng_;
        mutable std::mutex best_mutex_;
        std::mutex population_mutex_;
        int stagnation_count_ = 0;


        int vnd_max_adaptive_ = 20;


        std::atomic<bool> is_stuck_{ false };
        Island* ring_predecessor_{ nullptr };
        std::chrono::steady_clock::time_point last_migration_time_;
        std::chrono::steady_clock::time_point last_catastrophy_time_;
        std::chrono::steady_clock::time_point last_cross_island_pr_time_;
        std::chrono::steady_clock::time_point last_heavy_rr_time_;


        std::chrono::steady_clock::time_point intensification_end_time_;
        bool intensification_active_ = false;


        std::vector<Island*>
            exploit_siblings_;
        std::mutex broadcast_mutex_;
        std::vector<Individual>
            broadcast_buffer_;


        double last_broadcast_fitness_ = 1e30;
        std::vector<int> last_broadcast_genotype_;
        std::chrono::steady_clock::time_point last_broadcast_time_;

        void ProcessBroadcastBuffer();


        long long cache_hits_ = 0;
        long long cache_misses_ = 0;


        static constexpr int CACHE_WINDOW_SIZE = 1000;
        std::deque<bool> cache_result_window_;
        int cache_hits_in_window_ = 0;
        bool convergence_alarm_active_ = false;
        double convergence_mutation_boost_ = 1.0;

        void TrackCacheResult(bool was_hit);


        void OnConvergenceWarning();
        void OnConvergenceAlarm();
        void OnConvergenceCritical();


        long long diag_vnd_calls_ = 0;
        long long diag_vnd_improvements_ = 0;
        long long diag_mutations_ = 0;
        long long diag_strong_mutations_ = 0;
        long long diag_crossovers_ = 0;
        long long diag_offspring_better_ = 0;
        long long diag_offspring_total_ = 0;


        long long diag_srex_calls_ = 0;
        long long diag_srex_wins_ = 0;
        long long diag_neighbor_calls_ = 0;
        long long diag_neighbor_wins_ = 0;
        long long diag_pr_calls_ = 0;
        long long diag_pr_wins_ = 0;


        void LogDiagnostics(const std::chrono::steady_clock::time_point& now_diag);

        long long prof_mutation_time_us_ = 0;
        long long prof_vnd_time_us_ = 0;
        long long prof_eval_time_us_ = 0;
        long long prof_total_time_us_ = 0;
        long long prof_broadcast_time_us_ = 0;
        long long prof_succession_time_us_ = 0;

        long long prof_routepool_time_us_ = 0;
        long long prof_frankenstein_time_us_ = 0;
        long long prof_stagnation_time_us_ = 0;
        long long prof_diversity_time_us_ = 0;
        long long prof_offspring_time_us_ = 0;
        int prof_generations_ = 0;


        struct AdaptiveOperator {
            double success_rate = 0.5;
            long long calls = 0;
            long long wins = 0;
        };
        AdaptiveOperator adapt_swap_;
        AdaptiveOperator adapt_ejection_;
        AdaptiveOperator adapt_swap3_;
        AdaptiveOperator adapt_swap4_;

        static constexpr double ADAPT_ALPHA = 0.2;
        static constexpr double ADAPT_EPSILON =
            Config::ADAPT_EPSILON;

        void UpdateAdaptiveProbabilities();

        int SelectAdaptiveOperator();

        std::chrono::steady_clock::time_point last_diag_time_;


        std::chrono::steady_clock::time_point start_time_;
        std::chrono::steady_clock::time_point last_alns_print_time_;
        std::chrono::steady_clock::time_point last_greedy_assembly_time_;

        RoutePool route_pool_;
        Split split_;

        double max_diversity_baseline_ = 1.0;
        double min_diversity_baseline_ = 0.0;

        void CalibrateDiversity();
        void CalibrateConvergence();

        double current_structural_diversity_ = 0.0;
        double adaptive_mutation_rate_ = 0.0;
        double adaptive_vnd_prob_ = 0.0;
        double adaptive_ruin_chance_ = 0.0;


        double p_microsplit_ = 0.0;

        double p_retminimizer_ = 0.0;
        double p_mergesplit_ = 0.0;
        double p_swap3_ = 0.0;
        double p_swap4_ = 0.0;

        long long current_generation_ = 0;
        long long last_improvement_gen_ = 0;
        long long last_catastrophy_gen_ = 0;
        std::chrono::steady_clock::time_point
            immune_until_time_;

        std::chrono::steady_clock::time_point
            last_improvement_time_;




        std::chrono::steady_clock::time_point migration_lock_until_;

        int catastrophes_since_improvement_ = 0;

        void NuclearCatastrophe();

        const int BASE_STAGNATION_LIMIT = 1000;
        const int MAX_EXPLORATION_STAGNATION = 3000;



        mutable std::vector<int> pred1;
        mutable std::vector<int> pred2;
        mutable std::vector<int> last_in_group1;
        mutable std::vector<int> last_in_group2;

        void UpdateAdaptiveParameters();
        double MapRange(double value, double in_min, double in_max, double out_min,
            double out_max) const;


        int GetVndIterations() const;

        bool ShouldTrackDiversity() const {
            return current_generation_ % 10 == 0 ? true : false;
        }



        double SafeEvaluate(Individual& indiv);
        double SafeEvaluate(const std::vector<int>& genotype);

        void InitIndividual(Individual& indiv, INITIALIZATION_TYPE strategy);
 
        void InitIndividualKCenter(Individual& indiv);
        void LogStatus(long long time_since);
        void LogCatastropheTrigger(const char* reason, double unique_ratio, double health, double vnd_rate);
        void LogFrankensteinResult(double fitness, bool force_injected, bool improved_worst);
        void LogNuclearCatastrophe(bool start);
        void LogBroadcast(int recipient_id, double fitness);


        void UpdateBiasedFitness();
        void Catastrophy();
        int CalculateBrokenPairsDistance(const Individual& ind1,
            const Individual& ind2,
            const std::vector<int>& permutation,
            int num_groups);

        int CalculateSampledBPD(const Individual& ind1, const Individual& ind2,
            int sample_size = 400);

        int SelectParentIndex();
        int GetWorstBiasedIndex() const;
        int GetWorstIndex() const;

        Individual Crossover(const Individual& p1, const Individual& p2);

        Individual CrossoverNeighborBased(const Individual& p1, const Individual& p2);

        int ApplyMutation(Individual& child, bool is_endgame);
        int ApplyMicroSplitMutation(Individual& child);


        void ApplySplitToIndividual(Individual& indiv);
        Individual ApplySREX(const Individual& p1, const Individual& p2);

        void ApplySuccessionAdaptive(std::vector<Individual>& offspring_pool);

        void PerformCrossIslandPathRelinking();

        size_t last_routes_added_snapshot_ = 0;

    };

}