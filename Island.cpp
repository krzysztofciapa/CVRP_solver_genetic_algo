#define _USE_MATH_DEFINES
#include "Island.hpp"
#include "Constants.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_set>

using namespace LcVRPContest;
using namespace std;

static inline uint64_t HashGenotype64(const std::vector<int>& g) {
    uint64_t h = 1469598103934665603ULL;
    for (int x : g) {
        uint32_t v = static_cast<uint32_t>(x);
        h ^= (v & 0xFFu);
        h *= 1099511628211ULL;
        h ^= ((v >> 8) & 0xFFu);
        h *= 1099511628211ULL;
        h ^= ((v >> 16) & 0xFFu);
        h *= 1099511628211ULL;
        h ^= ((v >> 24) & 0xFFu);
        h *= 1099511628211ULL;
    }
    h ^= static_cast<uint64_t>(g.size()) + 0x9e3779b97f4a7c15ULL + (h << 6) +
        (h >> 2);
    return h;
}

Island::Island(ThreadSafeEvaluator* evaluator, const ProblemData& data,
    int population_size, int id)
    : evaluator_(evaluator), demands_(data.GetDemands()),
    capacity_(data.GetCapacity()), geometry_(data, id),
    local_search_(evaluator, &geometry_, id),
    population_size_(population_size), id_(id),
    current_best_(evaluator->GetSolutionSize()),
    rng_(static_cast<unsigned int>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() +
        id * 100000)),
    split_(evaluator) {
    population_.reserve(population_size_);
    local_cache_.InitHistory(evaluator->GetSolutionSize());

    int n = evaluator_->GetSolutionSize();
    int g = evaluator_->GetNumGroups();

    mutator_.Initialize(evaluator_, &geometry_, &split_);

    //bpd buffers
    size_t p_size = (size_t)n;
    size_t g_size = (size_t)g;
    pred1.resize(p_size);
    pred2.resize(p_size);
    last_in_group1.resize(g_size, -1);
    last_in_group2.resize(g_size, -1);

    int dim = data.GetDimension();
    customer_ranks_.resize(dim + 1, 0);
    const auto& perm = data.GetPermutation();
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] >= 0 && perm[i] < (int)customer_ranks_.size()) {
            customer_ranks_[perm[i]] = static_cast<int>(i);
        }
    }
    CalibrateDiversity(); // using monte carlo


    start_time_ = std::chrono::steady_clock::now();
    last_alns_print_time_ = std::chrono::steady_clock::now();
    last_greedy_assembly_time_ = std::chrono::steady_clock::now();

    //time based catastrophy counters
    last_migration_time_ = std::chrono::steady_clock::now();
    last_catastrophy_time_ = std::chrono::steady_clock::now();
    last_cross_island_pr_time_ = std::chrono::steady_clock::now();
    last_heavy_rr_time_ = std::chrono::steady_clock::now();
    last_improvement_time_ = std::chrono::steady_clock::now();
    last_broadcast_time_ = std::chrono::steady_clock::now();
}

double Island::EvaluateWithHistoryPenalty(const std::vector<int>& genotype) {
    double base_cost = SafeEvaluate(genotype);
    if (base_cost >= 1e20)
        return base_cost;

    double history_penalty = 0.0;
    double lambda = Config::HISTORY_LAMBDA;

    size_t size = genotype.size();
    if (size < 2)
        return base_cost;

    for (size_t i = 0; i < size - 1; ++i) {
        if (genotype[i] == genotype[i + 1]) {
            int freq = local_cache_.GetFrequency(i);
            history_penalty += freq * lambda;
        }
    }
    return base_cost + history_penalty;
}

double Island::SafeEvaluate(const std::vector<int>& genotype) {
    double distance = 0.0;
    int returns = 0;

    if (local_cache_.TryGet(genotype, distance, returns)) {
        cache_hits_++;
        TrackCacheResult(true); // rolling window tracking
        return distance;
    }
    cache_misses_++;
    TrackCacheResult(false); // rolling window tracking
    EvaluationResult result = evaluator_->EvaluateWithStats(genotype);

    distance = result.fitness;
    returns = result.returns;

    if (distance >= 1e9 || distance < 0.0) {
        distance = std::numeric_limits<double>::max();
    }
    local_cache_.Insert(genotype, distance, returns);

    return distance;
}

double Island::SafeEvaluate(Individual& indiv) {
    double distance = 0.0;
    int returns = 0;
    if (local_cache_.TryGet(indiv.AccessGenotype(), distance, returns)) {
        cache_hits_++;
        TrackCacheResult(true); // rolling window tracking
        indiv.SetReturnCount(returns);
        indiv.SetFitness(distance);
        return distance;
    }
    cache_misses_++;
    TrackCacheResult(false); // rolling window tracking;

    EvaluationResult result = evaluator_->EvaluateWithStats(indiv.GetGenotype());

    distance = result.fitness;
    returns = result.returns;

    if (distance >= 1e15 || distance < 0.0) {
        distance = std::numeric_limits<double>::max();
    }

    indiv.SetReturnCount(returns);
    indiv.SetFitness(distance); // raw fitness

    local_cache_.Insert(indiv.AccessGenotype(), distance, returns);

    return distance;
}

void Island::InitIndividual(Individual& indiv, INITIALIZATION_TYPE strategy) {
    int num_groups = evaluator_->GetNumGroups();
    std::vector<int>& genes = indiv.AccessGenotype();
    int num_clients = static_cast<int>(genes.size());

    if (strategy == INITIALIZATION_TYPE::K_CENTER_CLUSTERING) {
        InitIndividualKCenter(indiv);
        return;
    }

    std::vector<int> assignment_pool;
    assignment_pool.reserve(num_clients);
    for (int g = 0; g < num_groups; ++g) {
        assignment_pool.push_back(g);
    }
    int remaining_slots = num_clients - num_groups;

    if (remaining_slots > 0) {
        if (strategy == INITIALIZATION_TYPE::RR) {
            for (int i = 0; i < remaining_slots; ++i) {
                assignment_pool.push_back(i % num_groups);
            }
        }
        else if (strategy == INITIALIZATION_TYPE::CHUNKED) {
            int chunk_size =
                (num_groups > 0) ? (num_clients / num_groups) : num_clients;
            int current_group = 0;
            for (int i = 0; i < num_clients; ++i) {
                genes[i] = current_group;
                if ((i + 1) % chunk_size == 0 && current_group < num_groups - 1) {
                    current_group++;
                }
            }
            if (num_clients > 2) {
                std::uniform_int_distribution<int> d_swap(0, num_clients - 1);
                int a = d_swap(rng_);
                int b = d_swap(rng_);
                std::swap(genes[a], genes[b]);
            }
            return;
        }
        else {
            std::uniform_int_distribution<int> dist(0, num_groups - 1);
            for (int i = 0; i < remaining_slots; ++i) {
                assignment_pool.push_back(dist(rng_));
            }
        }
    }
    if ((int)assignment_pool.size() < num_clients) {
        assignment_pool.resize(num_clients, 0);
    }

    std::shuffle(assignment_pool.begin(), assignment_pool.end(), rng_);

    for (int i = 0; i < num_clients; ++i) {
        genes[i] = assignment_pool[i];
    }
}



void Island::InitIndividualKCenter(Individual& indiv) {



    int num_groups = evaluator_->GetNumGroups();
    int capacity = evaluator_->GetCapacity();
    std::vector<int>& genes = indiv.AccessGenotype();
    int num_clients = static_cast<int>(genes.size());
    const auto& perm = evaluator_->GetPermutation();

    //select k centers
    std::vector<int> centers;
    centers.reserve(num_groups);

    //first center: random
    int first_center_idx = rng_() % num_clients;
    centers.push_back(first_center_idx);


    std::vector<double> min_dist_to_center(num_clients, std::numeric_limits<double>::max());


    auto GetClientId = [&](int idx) { return idx + 2; }; //lambda for 0->2


    for (int i = 0; i < num_clients; ++i) {
        double d = evaluator_->GetDist(i + 1, first_center_idx + 1);
        min_dist_to_center[i] = d;
    }


    bool can_continue = true;
    for (int k = 1; k < num_groups && can_continue; ++k) {

        int best_candidate = -1;
        double max_d = -1.0;


        for (int i = 0; i < num_clients; ++i) {
            if (min_dist_to_center[i] > max_d) {
                max_d = min_dist_to_center[i];
                best_candidate = i;
            }
        }

        if (best_candidate != -1) {
            centers.push_back(best_candidate);

            for (int i = 0; i < num_clients; ++i) {
                double d = evaluator_->GetDist(i + 1, best_candidate + 1);
                if (d < min_dist_to_center[i]) {
                    min_dist_to_center[i] = d;
                }
            }
        }
        else {
            can_continue = false;
        }
    }

    //assign clients to centers with capacity consideration

    std::vector<int> group_loads(num_groups, 0);
    std::vector<int> client_indices(num_clients);
    std::iota(client_indices.begin(), client_indices.end(), 0);


    std::shuffle(client_indices.begin(), client_indices.end(), rng_);

    std::vector<std::pair<double, int>> center_options;
    center_options.reserve(num_groups);

    for (int client_idx : client_indices) {
        int demand = evaluator_->GetDemand(GetClientId(client_idx));

        center_options.clear();

        for (int g = 0; g < (int)centers.size(); ++g) {
            double d = evaluator_->GetDist(client_idx + 1, centers[g] + 1);
            center_options.push_back({ d, g });
        }

        std::sort(center_options.begin(), center_options.end());

        int assigned_group = -1;
        bool assigned = false;
        for (size_t i = 0; i < center_options.size() && !assigned; ++i) {
            int g = center_options[i].second;
            if (group_loads[g] + demand <= capacity) {
                assigned_group = g;
                assigned = true;
            }
        }

        if (assigned_group == -1) {
            int min_load = std::numeric_limits<int>::max();
            for (int g = 0; g < num_groups; ++g) {
                if (group_loads[g] < min_load) {
                    min_load = group_loads[g];
                    assigned_group = g;
                }
            }
        }

        genes[client_idx] = assigned_group;
        group_loads[assigned_group] += demand;
    }
}

void Island::Initialize(INITIALIZATION_TYPE strategy) {
    geometry_.Initialize(evaluator_);
    CalibrateDiversity();
    population_.clear();
    int sol_size = evaluator_->GetSolutionSize();
    Individual splitInd(sol_size);
    if (!Config::split) {
        InitIndividual(splitInd, INITIALIZATION_TYPE::RANDOM);
        population_.push_back(splitInd);
    }
    else {
        ApplySplitToIndividual(splitInd);
        double splitFit = SafeEvaluate(splitInd);
        splitInd.SetFitness(splitFit);
        population_.push_back(splitInd);
        {
            std::lock_guard<std::mutex> lock(best_mutex_);
            current_best_ = splitInd;
        }
    }

    //initialization ratios based on Island ID, switch bc num_islands is hardcoded

    double r_kcenter = 0.0, r_random = 0.0, r_chunked = 0.0;

    switch (id_) {
    case 0: r_kcenter = 0.50; r_random = 0.30; r_chunked = 0.20; break;
    case 1: r_kcenter = 0.60; r_random = 0.20; r_chunked = 0.20; break;
    case 2: r_kcenter = 0.30; r_random = 0.40; r_chunked = 0.30; break;
    case 3: r_kcenter = 0.30; r_random = 0.20; r_chunked = 0.50; break;
    case 4: r_kcenter = 0.20; r_random = 0.60; r_chunked = 0.20; break;
    case 5: r_kcenter = 0.35; r_random = 0.35; r_chunked = 0.30; break;
    default: r_kcenter = 0.40; r_random = 0.30; r_chunked = 0.30; break;
    }

    int k_center_count = static_cast<int>(population_size_ * r_kcenter);
    int random_count = static_cast<int>(population_size_ * r_random);
    int chunked_count = static_cast<int>(population_size_ * r_chunked);

    int current_total = k_center_count + random_count + chunked_count;
    if (current_total < population_size_) {
        random_count += (population_size_ - current_total);
    }

    int zero_return_count = 0;

    for (int i = 0; i < population_size_; ++i) {
        Individual indiv(sol_size);

        INITIALIZATION_TYPE type;
        if (i < k_center_count) {
            type = INITIALIZATION_TYPE::K_CENTER_CLUSTERING;
        }
        else if (i < k_center_count + chunked_count) {
            type = INITIALIZATION_TYPE::CHUNKED;
        }
        else {
            type = INITIALIZATION_TYPE::RANDOM;
        }

        InitIndividual(indiv, type);


        double fit = SafeEvaluate(indiv);

        if (indiv.GetReturnCount() == 0)
            zero_return_count++;


        if ((type == INITIALIZATION_TYPE::K_CENTER_CLUSTERING || type == INITIALIZATION_TYPE::CHUNKED)
            && indiv.GetReturnCount() == 0) {
            //if has 0 returns, try to optimize with VND
            local_search_.RunVND(indiv);

            fit = SafeEvaluate(indiv);
        }

        indiv.SetFitness(fit);
        population_.push_back(indiv);

        if (fit < current_best_.GetFitness()) {
            std::lock_guard<std::mutex> lock(best_mutex_);
            current_best_ = indiv;
        }
    }

    if (Config::SHOW_LOGS) {
        std::cout << "[ISLAND " << id_
            << "] Initialized. Ratios: K=" << (int)(r_kcenter * 100)
            << "% R=" << (int)(r_random * 100) << "% C=" << (int)(r_chunked * 100) << "%"
            << " | 0-Ret: " << zero_return_count << "/" << population_size_ << "\n";
    }

    stagnation_count_ = 0;
    catastrophes_since_improvement_ = 0;
    migration_lock_until_ = std::chrono::steady_clock::time_point();

    UpdateBiasedFitness();
}

void Island::RunGeneration() {
    std::chrono::high_resolution_clock::time_point t_gen_start, t_bdc_start;
    if (Config::SHOW_LOGS) {
        t_gen_start = std::chrono::high_resolution_clock::now();
        t_bdc_start = std::chrono::high_resolution_clock::now();
    }

    ProcessBroadcastBuffer(); // process broadcasts

    if (Config::SHOW_LOGS) {
        auto t_bdc_end = std::chrono::high_resolution_clock::now();
        prof_broadcast_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_bdc_end - t_bdc_start).count();
    }

    current_generation_++;

    //check if intensification phase is over
    if (intensification_active_) {
        auto now_check = std::chrono::steady_clock::now();
        if (now_check > intensification_end_time_) {
            intensification_active_ = false;
        }
    }

    if (ShouldTrackDiversity()) {
        UpdateBiasedFitness();
        UpdateAdaptiveParameters();
    }
    stagnation_count_++;


    auto now_diag = std::chrono::steady_clock::now();
    double since_last_diag =
        std::chrono::duration<double>(now_diag - last_diag_time_).count();
    if (since_last_diag >= 30.0 && current_generation_ > 10) {
        last_diag_time_ = now_diag;
        LogDiagnostics(now_diag);


        diag_vnd_calls_ = diag_vnd_improvements_ = 0;
        if (Config::SHOW_LOGS) {
            diag_mutations_ = diag_strong_mutations_ = 0;
            diag_crossovers_ = diag_offspring_better_ = diag_offspring_total_ = 0;
            diag_srex_calls_ = diag_srex_wins_ = 0;
            diag_neighbor_calls_ = diag_neighbor_wins_ = 0;
            diag_pr_calls_ = diag_pr_wins_ = 0;


            prof_mutation_time_us_ = 0;
            prof_vnd_time_us_ = 0;
            prof_eval_time_us_ = 0;
            prof_total_time_us_ = 0;
            prof_broadcast_time_us_ = 0;
            prof_succession_time_us_ = 0;
            prof_routepool_time_us_ = 0;
            prof_frankenstein_time_us_ = 0;
            prof_stagnation_time_us_ = 0;
            prof_diversity_time_us_ = 0;
            prof_offspring_time_us_ = 0;
            prof_generations_ = 0;
        }


        if (IsExploitation()) {
            UpdateAdaptiveProbabilities();
        }


        double recent_hit_rate = GetRecentCacheHitRate();
        if (cache_result_window_.size() >= CACHE_WINDOW_SIZE / 2) {
            if (recent_hit_rate > 0.95 && !convergence_alarm_active_) {
                OnConvergenceCritical();
            }
            else if (recent_hit_rate > 0.90 && !convergence_alarm_active_) {
                OnConvergenceAlarm();
            }
            else if (recent_hit_rate > 0.85 && convergence_mutation_boost_ < 2.0) {
                OnConvergenceWarning();
            }
            else if (recent_hit_rate < 0.70) {
                convergence_mutation_boost_ = 1.0;
                convergence_alarm_active_ = false;
            }
        }
    }

    //asnynchronous migration pull
    TryPullMigrant();

    //perform cross-island path relinking
    if (IsExploitation()) {
        PerformCrossIslandPathRelinking();
    }

    const int lambda = population_size_;
    std::vector<Individual> offspring_pool;
    offspring_pool.reserve(lambda);

    double fitness_threshold = std::numeric_limits<double>::max();
    if (!population_.empty()) {
        fitness_threshold = population_[population_.size() / 2].GetFitness();
    }

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();
    bool is_endgame =
        (elapsed > Config::MAX_TIME_SECONDS * Config::ENDGAME_THRESHOLD);


    std::uniform_real_distribution<double> d(0.0, 1.0);
    std::uniform_real_distribution<double> dist_op(0.0, 1.0);

    for (int i = 0; i < lambda; ++i) {
        Individual child(evaluator_->GetSolutionSize());
        int p1 = -1, p2 = -1;
        double op_val = dist_op(rng_);

        bool mutated = false;
        bool strong_mutation = false;
        int op_type = 0; // 0=None, 1=SREX, 2=Neighbor, 3=PR, 4=Mutation(RR/Split)
        int crossover_type = 0;
        double parent1_fit = 0.0;
        double parent2_fit = 0.0;
        bool operator_selected = false;

        if (IsExploitation()) {

            int problem_size_rr = evaluator_->GetSolutionSize();
            bool should_apply_rr = false;
            double rr_intensity = Config::EXPLOIT_HEAVY_RR_INTENSITY;

            if (problem_size_rr > Config::LARGE_INSTANCE_THRESHOLD) {

                auto now_rr = std::chrono::steady_clock::now();
                double seconds_stagnant =
                    std::chrono::duration<double>(now_rr - last_improvement_time_).count();
                double seconds_since_rr =
                    std::chrono::duration<double>(now_rr - last_heavy_rr_time_).count();
                double health_rr = GetPopulationHealth();

                should_apply_rr =
                    (seconds_stagnant > Config::EXPLOIT_RR_STAGNATION_SECONDS ||
                        health_rr < Config::EXPLOIT_RR_HEALTH_THRESHOLD) &&
                    seconds_since_rr > Config::EXPLOIT_RR_INTERVAL_SECONDS &&
                    i == 0;

                if (should_apply_rr) {
                    rr_intensity = (seconds_stagnant > Config::EXPLOIT_EXTREME_STAGNATION_SECONDS)
                        ? Config::EXPLOIT_RR_INTENSITY_EXTREME
                        : Config::EXPLOIT_HEAVY_RR_INTENSITY;
                    last_heavy_rr_time_ = std::chrono::steady_clock::now();
                }
            }
            else {
                //for small instances use generation based logic
                long long time_since_improve = current_generation_ - last_improvement_gen_;
                should_apply_rr =
                    time_since_improve > Config::EXPLOIT_RR_STAGNATION_TRIGGER &&
                    i == 0 &&
                    current_generation_ % Config::EXPLOIT_RR_INTERVAL == 0;
            }

            if (should_apply_rr) {
                int victim = rng_() % population_.size();
                child = population_[victim];

                //adaptive lns for large, 90% simple RR and 10% adaptive lns for small
                bool use_adaptive = (problem_size_rr > Config::LARGE_INSTANCE_THRESHOLD);

                if (!use_adaptive) {

                    if (std::uniform_real_distribution<double>(0.0, 1.0)(rng_) < 0.10) {
                        use_adaptive = true;
                    }
                }

                if (use_adaptive) {
                    LNSStrategy strategy = aos_.SelectLNSStrategy(rng_);
                    mutator_.ApplyAdaptiveLNS(child, strategy, rr_intensity, true, rng_);
                }
                else {
                    mutator_.ApplyRuinRecreate(child, rr_intensity, true, rng_);
                }

                mutated = true;
                strong_mutation = true;
                op_type = 4;
                operator_selected = true;
            }

            if (!operator_selected) {

                double pr_prob, rr_prob, split_prob;
                //specializations for exploit
                switch (id_) {
                case 1:
                    pr_prob = Config::EXPLOIT_I1_OP_PR_PROB;
                    rr_prob = Config::EXPLOIT_I1_OP_RR_PROB;
                    split_prob = Config::EXPLOIT_I1_OP_SPLIT_PROB;
                    break;
                case 3:
                    pr_prob = Config::EXPLOIT_I3_OP_PR_PROB;
                    rr_prob = Config::EXPLOIT_I3_OP_RR_PROB;
                    split_prob = Config::EXPLOIT_I3_OP_SPLIT_PROB;
                    break;
                case 5:
                    pr_prob = Config::EXPLOIT_I5_OP_PR_PROB;
                    rr_prob = Config::EXPLOIT_I5_OP_RR_PROB;
                    split_prob = Config::EXPLOIT_I5_OP_SPLIT_PROB;
                    break;
                default:
                    pr_prob = 0.60;
                    rr_prob = 0.30;
                    split_prob = 0.10;
                    break;
                }


                if (evaluator_->GetSolutionSize() > Config::HUGE_INSTANCE_THRESHOLD) {
                    pr_prob = 0.30;
                    rr_prob = 0.50;
                    split_prob = 0.20;
                }

                if (op_val < pr_prob) {

                    int candidates[4];
                    for (int c = 0; c < 4; ++c)
                        candidates[c] = SelectParentIndex();

                    int best_c1 = 0, best_c2 = 1;
                    int max_dist = -1;
                    int num_groups = evaluator_->GetNumGroups();

                    for (int c1 = 0; c1 < 4; ++c1) {
                        for (int c2 = c1 + 1; c2 < 4; ++c2) {
                            int d = CalculateBrokenPairsDistance(
                                population_[candidates[c1]], population_[candidates[c2]],
                                evaluator_->GetProblemData().GetPermutation(), num_groups);
                            if (d > max_dist) {
                                max_dist = d;
                                best_c1 = candidates[c1];
                                best_c2 = candidates[c2];
                            }
                        }
                    }
                    p1 = best_c1;
                    p2 = best_c2;
                    parent1_fit = population_[p1].GetFitness();
                    parent2_fit = population_[p2].GetFitness();

                    int problem_size = evaluator_->GetSolutionSize();
                    if (problem_size > Config::HUGE_INSTANCE_THRESHOLD) {

                        child = CrossoverNeighborBased(population_[p1], population_[p2]);
                        op_type = 1;
                        crossover_type = 2;
                    }
                    else {
                        if (parent1_fit < parent2_fit) {
                            child = population_[p1];
                            double cost = child.GetFitness();
                            auto t_pr_start = std::chrono::high_resolution_clock::now();
                            local_search_.TryPathRelinking(child.AccessGenotype(), cost,
                                population_[p2].GetGenotype());
                            auto t_pr_end = std::chrono::high_resolution_clock::now();

                            if (Config::SHOW_LOGS) prof_vnd_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_pr_end - t_pr_start).count();
                            child.SetFitness(cost);
                        }
                        else {
                            child = population_[p2];
                            double cost = child.GetFitness();
                            auto t_pr_start = std::chrono::high_resolution_clock::now();
                            local_search_.TryPathRelinking(child.AccessGenotype(), cost,
                                population_[p1].GetGenotype());
                            auto t_pr_end = std::chrono::high_resolution_clock::now();
                            if (Config::SHOW_LOGS) prof_vnd_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_pr_end - t_pr_start).count();
                            child.SetFitness(cost);
                        }
                        op_type = 3;// pr
                        crossover_type = 3;
                    }

                }
                else if (op_val < pr_prob + rr_prob) {

                    p1 = SelectParentIndex();
                    child = population_[p1];

                    auto t_mut_start = std::chrono::high_resolution_clock::now();
                    mutator_.ApplyRuinRecreate(child, 0.0, true, rng_);
                    auto t_mut_end = std::chrono::high_resolution_clock::now();
                    if (Config::SHOW_LOGS) prof_vnd_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_mut_end - t_mut_start).count();
                    mutated = true;
                    op_type = 4;
                }
                else {

                    p1 = SelectParentIndex();
                    child = population_[p1];

                    auto t_mut_start = std::chrono::high_resolution_clock::now();
                    mutator_.ApplyMicroSplitMutation(child, 0.0, 2, rng_);
                    auto t_mut_end = std::chrono::high_resolution_clock::now();
                    if (Config::SHOW_LOGS) prof_vnd_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_mut_end - t_mut_start).count();
                    mutated = true;
                    op_type = 4;
                }
            }

        }
        else {

            double p_crossover = 0.60;

            // adaptive crossover prob
            p_crossover = 0.40 + (0.40 * current_structural_diversity_);
            if (Config::SHOW_LOGS && current_generation_ % 100 == 0 && i == 0) {

            }


            p1 = SelectParentIndex();
            p2 = SelectParentIndex();

            if (p1 >= 0 && p2 >= 0 && op_val < p_crossover) {

                parent1_fit = population_[p1].GetFitness();
                parent2_fit = population_[p2].GetFitness();

                if (is_endgame) {
                    auto t_xo_start = std::chrono::high_resolution_clock::now();
                    child = CrossoverNeighborBased(population_[p1], population_[p2]);
                    auto t_xo_end = std::chrono::high_resolution_clock::now();
                    if (Config::SHOW_LOGS) prof_vnd_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_xo_end - t_xo_start).count();
                    crossover_type = 2;
                }
                else {

                    double seq_prob = (id_ <= 1) ? 0.31 : 0.5;
                    crossover_type = (dist_op(rng_) < seq_prob) ? 1 : 2;
                    auto t_xo_start = std::chrono::high_resolution_clock::now();
                    child = Crossover(population_[p1], population_[p2]);
                    auto t_xo_end = std::chrono::high_resolution_clock::now();
                    if (Config::SHOW_LOGS) prof_vnd_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_xo_end - t_xo_start).count();
                }
            }
            else {

                if (p1 >= 0)
                    child = population_[p1];
                else
                    InitIndividual(child, INITIALIZATION_TYPE::RANDOM);

                int m_res = ApplyMutation(child, is_endgame);
                mutated = (m_res > 0);
                strong_mutation = (m_res == 2);
            }
        }


        if (Config::SHOW_LOGS) {
            if (mutated)
                diag_mutations_++;
            if (strong_mutation)
                diag_strong_mutations_++;
            if (crossover_type > 0)
                diag_crossovers_++;
        }


        int problem_size = evaluator_->GetSolutionSize();

        child.Canonicalize();
        double fit = 0;
        int ret = 0;
        if (!local_cache_.TryGet(child.GetGenotype(), fit, ret)) {
            cache_misses_++;
            EvaluationResult res = evaluator_->EvaluateWithStats(child.GetGenotype());
            fit = res.fitness;
            ret = res.returns;
            local_cache_.Insert(child.GetGenotype(), fit, ret);
        }
        else {
            cache_hits_++;
        }
        child.SetFitness(fit);
        child.SetReturnCount(ret);


        if (Config::SHOW_LOGS && crossover_type > 0 && parent1_fit > 0 && parent2_fit > 0) {
            if (fit < parent1_fit && fit < parent2_fit) {
                if (crossover_type == 1)
                    diag_srex_wins_++;
                else if (crossover_type == 2)
                    diag_neighbor_wins_++;
                else if (crossover_type == 3)
                    diag_pr_wins_++;
            }
            if (crossover_type == 1)
                diag_srex_calls_++;
            else if (crossover_type == 2)
                diag_neighbor_calls_++;
            else if (crossover_type == 3)
                diag_pr_calls_++;
        }

        bool promising = (fit < fitness_threshold);


        double vnd_prob;
        if (problem_size > Config::LARGE_INSTANCE_THRESHOLD) {
            vnd_prob = IsExploration() ? Config::EXPLORATION_VND_PROB_LARGE
                : Config::EXPLOITATION_VND_PROB_LARGE;
        }
        else {
            vnd_prob = IsExploration() ? Config::EXPLORATION_VND_PROB
                : Config::EXPLOITATION_VND_PROB;
        }

        bool exploration_vnd =
            IsExploration() &&
            (promising || (d(rng_) < Config::EXPLORATION_VND_EXTRA_PROB));


        bool should_run_vnd = exploration_vnd || strong_mutation ||
            (IsExploitation() && promising) ||
            (d(rng_) < vnd_prob) || is_endgame || intensification_active_;



        if (should_run_vnd) {
            int vnd_iters = GetVndIterations();
            if (intensification_active_) {
                vnd_iters = Config::EXPLOITATION_VND_MAX;
            }
            else if (is_endgame) {
                vnd_iters = Config::EXPLOITATION_VND_MAX;
            }
            else if (current_structural_diversity_ > 0.6) {
                vnd_iters = (int)(vnd_iters * 1.5);
            }


            if (problem_size > Config::LARGE_INSTANCE_THRESHOLD) {
                if (IsExploration()) {
                    vnd_iters = std::min(3, vnd_iters);
                }
                else if (problem_size > 2000) {

                    vnd_iters = std::min(vnd_iters, vnd_max_adaptive_);
                }
            }

            bool allow_swap = IsExploitation() && Config::ALLOW_SWAP;


            bool allow_3swap = false;
            bool allow_ejection = false;
            bool allow_4swap = false;


            if (problem_size > 3000 && IsExploration()) {
                allow_swap = false;
                allow_3swap = false;
                allow_ejection = false;
                allow_4swap = false;
            }

            int selected_op = -1;

            if (IsExploitation() && problem_size < Config::LARGE_INSTANCE_THRESHOLD &&
                !strong_mutation) {

                selected_op =
                    SelectAdaptiveOperator(); // 0=Swap, 1=Ejection, 2=3-Swap, 3=4-Swap

                allow_swap = false;


                switch (selected_op) {
                case 0:
                    allow_swap = true;
                    allow_ejection = true;
                    adapt_swap_.calls++;
                    break;
                case 1:
                    allow_ejection = true;
                    allow_3swap = true;
                    adapt_ejection_.calls++;
                    break;
                case 2:
                    allow_ejection = true;
                    allow_3swap = true;
                    allow_4swap = true;
                    adapt_swap3_.calls++;
                    break;
                case 3:
                    allow_3swap = true;
                    allow_4swap = true;
                    adapt_swap4_.calls++;
                    break;
                }
                // path relinking!
                local_search_.SetGuideSolution(current_best_.GetGenotype());
            }
            else if (IsExploration()) {

                if (problem_size <= 500) {

                    allow_swap = (d(rng_) < 0.50);
                    allow_ejection = false;

                }
                else if (problem_size <= Config::LARGE_INSTANCE_THRESHOLD) {

                    allow_swap = (d(rng_) < 0.50);
                    allow_ejection = false;
                }
                else {

                    allow_swap = false;
                    allow_3swap = false;
                    allow_ejection = false;
                }
                local_search_.SetGuideSolution({});
            }

            diag_vnd_calls_++;
            double fit_before = child.GetFitness();

            double full_vnd_prob = 0.10;
            if (problem_size > Config::LARGE_INSTANCE_THRESHOLD) {
                full_vnd_prob = (stagnation_count_ > 50) ? 0.50 : 0.25;
            }
            bool force_full_vnd = (IsExploitation() && (d(rng_) < full_vnd_prob)) || intensification_active_;
            if (force_full_vnd) {
                allow_swap = true;
                allow_3swap = true;
                allow_ejection = true;
            }


            if (problem_size > Config::LARGE_INSTANCE_THRESHOLD) {
                if (IsExploitation()) {
                    if (problem_size > 3000) {

                    }
                }
                else {

                    allow_3swap = false;
                    allow_4swap = false;
                    if (problem_size > 3000) {
                        allow_ejection = false;
                        allow_swap = false;
                    }
                }
            }


            bool vnd_improved = false;
            auto t_vnd_start = std::chrono::high_resolution_clock::now();


            if (problem_size > Config::HUGE_INSTANCE_THRESHOLD) {

                int tier;
                if (is_endgame) {
                    tier = 2;
                }
                else if (IsExploration()) {
                    tier = 0;
                }
                else if (current_generation_ % 50 == 0) {
                    tier = 2;
                }
                else {
                    tier = 1;
                }
                vnd_improved = local_search_.RunHugeInstanceVND(child, tier);
            }
            else if (problem_size > Config::LARGE_INSTANCE_THRESHOLD && !force_full_vnd) {
                vnd_improved =
                    local_search_.RunVND(child, vnd_iters, allow_swap, allow_3swap,
                        allow_ejection, allow_4swap);
            }
            else {

                vnd_improved =
                    local_search_.RunVND(child, vnd_iters, allow_swap, allow_3swap,
                        allow_ejection, allow_4swap);
            }
            auto t_vnd_end = std::chrono::high_resolution_clock::now();
            long long vnd_micros = std::chrono::duration_cast<std::chrono::microseconds>(t_vnd_end - t_vnd_start).count();
            if (Config::SHOW_LOGS) prof_vnd_time_us_ += vnd_micros;


            if (IsExploitation() && problem_size > 2000) {
                long long vnd_ms = vnd_micros / 1000;
                if (vnd_ms > 200) {

                    vnd_max_adaptive_ = std::max(1, vnd_max_adaptive_ / 2);
                }
                else if (vnd_ms < 50) {

                    vnd_max_adaptive_ = std::min(50, vnd_max_adaptive_ + 1);
                }
            }

            if (vnd_improved) {
                child.Canonicalize();
                if (!local_cache_.TryGet(child.GetGenotype(), fit, ret)) {
                    cache_misses_++;
                    auto t_eval_start = std::chrono::high_resolution_clock::now();
                    EvaluationResult res =
                        evaluator_->EvaluateWithStats(child.GetGenotype());
                    auto t_eval_end = std::chrono::high_resolution_clock::now();
                    if (Config::SHOW_LOGS) prof_eval_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_eval_end - t_eval_start).count();

                    fit = res.fitness;
                    ret = res.returns;
                    local_cache_.Insert(child.GetGenotype(), fit, ret);
                }
                else {
                    cache_hits_++;
                }
                child.SetFitness(fit);
                child.SetReturnCount(ret);
                if (fit < fit_before) {
                    diag_vnd_improvements_++;


                    if (IsExploitation() &&
                        problem_size < Config::LARGE_INSTANCE_THRESHOLD &&
                        selected_op >= 0) {
                        switch (selected_op) {
                        case 0:
                            adapt_swap_.wins++;
                            break;
                        case 1:
                            adapt_ejection_.wins++;
                            break;
                        case 2:
                            adapt_swap3_.wins++;
                            break;
                        case 3:
                            adapt_swap4_.wins++;
                            break;
                        }
                    }
                }
            }
        }


        if (Config::SHOW_LOGS) {
            diag_offspring_total_++;
            if (fit < fitness_threshold)
                diag_offspring_better_++;
        }

        offspring_pool.push_back(std::move(child));


        bool should_broadcast = false;
        Individual best_to_broadcast;

        {
            std::lock_guard<std::mutex> lock(best_mutex_);
            if (fit < current_best_.GetFitness()) {

                offspring_pool.back().SetNative(true);
                offspring_pool.back().SetHomeIsland(id_);

                current_best_ = offspring_pool.back();
                stagnation_count_ = 0;
                last_improvement_gen_ = current_generation_;
                last_improvement_time_ = std::chrono::steady_clock::now();

                //if we find new best, we intensify
                double elapsed_sec = std::chrono::duration<double>(last_improvement_time_ - start_time_).count();
                if (elapsed_sec > 30.0) {
                    auto now = std::chrono::steady_clock::now();
                    if (intensification_active_) {

                        intensification_end_time_ = intensification_end_time_ + std::chrono::seconds(3);
                    }
                    else {

                        intensification_active_ = true;
                        intensification_end_time_ = now + std::chrono::seconds(3);
                    }
                }

                fitness_threshold = fit * 1.05;

                if (!exploit_siblings_.empty()) {
                    should_broadcast = true;
                    best_to_broadcast = current_best_;
                }

                catastrophes_since_improvement_ = 0;
            }
        }

        auto now_broadcast = std::chrono::steady_clock::now();
        double elapsed_broadcast =
            std::chrono::duration<double>(now_broadcast - start_time_).count();

        if (should_broadcast) {
            auto t_bdc_start = std::chrono::high_resolution_clock::now();
            if (IsExploration()) {

                if (problem_size <= Config::LARGE_INSTANCE_THRESHOLD) {
                    best_to_broadcast.SetHomeIsland(id_);
                    for (Island* sibling : exploit_siblings_) {
                        if (sibling != nullptr) {
                            sibling->ReceiveBroadcastBest(best_to_broadcast);
                        }
                    }
                }
                else {

                    double BROADCAST_WARMUP_SECONDS = 10.0;
                    double BROADCAST_COOLDOWN_SECONDS = 5.0;

                    bool in_warmup = (elapsed_broadcast < BROADCAST_WARMUP_SECONDS);
                    auto now_bc = std::chrono::steady_clock::now();
                    double seconds_since_last_broadcast =
                        std::chrono::duration<double>(now_bc - last_broadcast_time_).count();
                    bool cooldown_passed = (seconds_since_last_broadcast >= BROADCAST_COOLDOWN_SECONDS);

                    if (!in_warmup && cooldown_passed) {
                        double improvement_ratio = 1.0;
                        if (last_broadcast_fitness_ < 1e20) {
                            improvement_ratio = (last_broadcast_fitness_ - best_to_broadcast.GetFitness())
                                / last_broadcast_fitness_;
                        }

                        bool significant_improvement = (improvement_ratio > 0.005);
                        bool too_small_improvement = (improvement_ratio < 0.001);

                        int bpd_to_last = 0;
                        if (!last_broadcast_genotype_.empty()) {
                            Individual temp_last(last_broadcast_genotype_);
                            bpd_to_last = CalculateBrokenPairsDistancePublic(best_to_broadcast, temp_last);
                        }
                        else {
                            bpd_to_last = evaluator_->GetSolutionSize();
                        }

                        int min_bpd_for_broadcast = evaluator_->GetSolutionSize() * 0.10;
                        bool structurally_different = (bpd_to_last > min_bpd_for_broadcast);

                        bool should_actually_broadcast =
                            (significant_improvement || structurally_different) && !too_small_improvement;

                        if (should_actually_broadcast) {
                            best_to_broadcast.SetHomeIsland(id_);
                            for (Island* sibling : exploit_siblings_) {
                                if (sibling != nullptr) {
                                    sibling->ReceiveBroadcastBest(best_to_broadcast);
                                }
                            }

                            last_broadcast_fitness_ = best_to_broadcast.GetFitness();
                            last_broadcast_genotype_ = best_to_broadcast.GetGenotype();
                            last_broadcast_time_ = now_bc;

                            if (Config::SHOW_LOGS) {
                                std::cout << "\033[36m [I" << id_ << " EXPLORE] Broadcast to EXPLOIT (imp=" << std::fixed << std::setprecision(2) << (improvement_ratio * 100)
                                    << "%, BPD=" << bpd_to_last << ")\033[0m\n";
                            }

                        }
                    }
                }
            }
            else {

                bool broadcast_enabled =
                    (elapsed_broadcast > Config::BROADCAST_WARMUP_SECONDS);
                if (broadcast_enabled) {
                    best_to_broadcast.SetHomeIsland(id_);
                    for (Island* sibling : exploit_siblings_) {
                        if (sibling != nullptr) {
                            sibling->ReceiveBroadcastBest(best_to_broadcast);
                        }
                    }
                }
            }
            auto t_bdc_end = std::chrono::high_resolution_clock::now();
            if (Config::SHOW_LOGS) prof_broadcast_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_bdc_end - t_bdc_start).count();
        }
    }
    if (Config::SHOW_LOGS) prof_generations_++;
    auto t_gen_end = std::chrono::high_resolution_clock::now();
    if (Config::SHOW_LOGS) prof_total_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_gen_end - t_gen_start).count();


    auto t_succ_start = std::chrono::high_resolution_clock::now();
    ApplySuccessionAdaptive(offspring_pool);
    auto t_succ_end = std::chrono::high_resolution_clock::now();
    if (Config::SHOW_LOGS) prof_succession_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_succ_end - t_succ_start).count();
    {
        std::lock_guard<std::mutex> lock(population_mutex_);
        for (auto& ind : population_)
            ind.IncrementStagnation();
    }

    long long time_since = current_generation_ - last_improvement_gen_;

    double worst_fit = 0.0;
    {
        std::lock_guard<std::mutex> lock(population_mutex_);
        for (const auto& ind : population_)
            if (ind.GetFitness() > worst_fit)
                worst_fit = ind.GetFitness();
    }

    double best_fit_for_catastrophe;
    {
        std::lock_guard<std::mutex> lock(best_mutex_);
        best_fit_for_catastrophe = current_best_.GetFitness();
    }


    int cat_problem_size = evaluator_->GetSolutionSize();


    if ((current_generation_ % 20 == 0 || current_structural_diversity_ == 0.0) && cat_problem_size <= Config::LARGE_INSTANCE_THRESHOLD) {
        int unique_count = 0;
        {
            std::lock_guard<std::mutex> lock(population_mutex_);
            std::vector<bool> is_unique(population_.size(), true);
            const auto& perm = evaluator_->GetPermutation();
            int num_groups = evaluator_->GetNumGroups();
            int min_bpd = cat_problem_size * 0.03;

            for (size_t i = 0; i < population_.size() && unique_count < (int)population_.size() / 2; ++i) {
                if (!is_unique[i]) continue;
                unique_count++;
                for (size_t j = i + 1; j < population_.size(); ++j) {
                    if (!is_unique[j]) continue;
                    int bpd = CalculateBrokenPairsDistance(population_[i], population_[j], perm, num_groups);
                    if (bpd < min_bpd) {
                        is_unique[j] = false;
                    }
                }
            }
        }
        current_structural_diversity_ = (double)unique_count / population_.size();
    }
    else if (cat_problem_size > Config::LARGE_INSTANCE_THRESHOLD) {
        current_structural_diversity_ = 1.0;
    }

    double unique_ratio = current_structural_diversity_;
    bool population_degenerated = (unique_ratio < 0.30);


    double health = GetPopulationHealth();
    bool health_critical = (health < Config::CATASTROPHE_HEALTH_THRESHOLD);

    double vnd_success_rate =
        (diag_vnd_calls_ > 0) ? (100.0 * diag_vnd_improvements_ / diag_vnd_calls_)
        : 100.0;
    bool vnd_exhausted = (vnd_success_rate < 2.0 && diag_vnd_calls_ > 500);


    bool stagnation_critical = IsExploration() && (stagnation_count_ > Config::STAGNATION_THRESHOLD * 10); // ~3000 generations

    bool should_catastrophe = false;

    if (cat_problem_size > Config::LARGE_INSTANCE_THRESHOLD) {

        auto now_cat = std::chrono::steady_clock::now();
        double seconds_since_cat =
            std::chrono::duration<double>(now_cat - last_catastrophy_time_).count();
        double seconds_since_improvement =
            std::chrono::duration<double>(now_cat - last_improvement_time_).count();

        bool in_grace_period = (seconds_since_improvement < 15.0);

        double dynamic_cooldown = (seconds_since_improvement < 120.0) ? 180.0 : 90.0;

        should_catastrophe = (population_degenerated || health_critical || vnd_exhausted) &&
            seconds_since_cat > dynamic_cooldown &&
            !(population_degenerated && in_grace_period);

        if (should_catastrophe) {
            last_catastrophy_time_ = now_cat;
        }
    }
    else {

        long long gens_since_cat = current_generation_ - last_catastrophy_gen_;
        long long gens_since_improve = current_generation_ - last_improvement_gen_;


        bool in_grace_period = (gens_since_improve < 300);

        should_catastrophe = (population_degenerated || health_critical || vnd_exhausted) &&
            gens_since_cat > Config::CATASTROPHE_MIN_GAP_GENS &&
            !(population_degenerated && in_grace_period);

        if (should_catastrophe) {
            last_catastrophy_gen_ = current_generation_;
        }
    }

    if (should_catastrophe || stagnation_critical) {
        catastrophes_since_improvement_++;
        const char* reason = population_degenerated ? "POP_DEGENERATED"
            : health_critical ? "HEALTH_CRITICAL"
            : stagnation_critical ? "STAGNATION_CRITICAL"
            : "VND_EXHAUSTED";


        if (IsExploration() && catastrophes_since_improvement_ > 3) {

            NuclearCatastrophe();
            catastrophes_since_improvement_ = 0;
        }
        else {
            LogCatastropheTrigger(reason, unique_ratio, health, vnd_success_rate);
            Catastrophy();
        }
    }


    is_stuck_.store(health < Config::HEALTH_SICK || vnd_exhausted,
        std::memory_order_relaxed);

    if (IsExploitation()) {
        {
            std::lock_guard<std::mutex> lock(best_mutex_);

            bool should_add_routes = (stagnation_count_ == 0 || current_generation_ % 100 == 0);
            if (evaluator_->GetSolutionSize() > Config::HUGE_INSTANCE_THRESHOLD) {
                should_add_routes = (stagnation_count_ == 0 && current_generation_ % 10 == 0) || (current_generation_ % 500 == 0);
            }

            if (should_add_routes) {
                auto t_rp_start = std::chrono::high_resolution_clock::now();
                route_pool_.AddRoutesFromSolution(current_best_.GetGenotype(), *evaluator_);
                auto t_rp_end = std::chrono::high_resolution_clock::now();
                if (Config::SHOW_LOGS) prof_routepool_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_rp_end - t_rp_start).count();
            }
        }

        size_t current_updates = route_pool_.GetTotalRoutesAdded();
        bool pool_updated = (current_updates > last_routes_added_snapshot_);


        auto now_gen = std::chrono::steady_clock::now();
        double elapsed_gen =
            std::chrono::duration<double>(now_gen - start_time_).count();

        int vnd_iters = GetVndIterations();
        if (is_endgame) {

            vnd_iters = Config::EXPLOITATION_VND_MAX;
        }

        //frankenstein
        bool use_frankenstein = Config::ENABLE_FRANKENSTEIN;

        if (IsExploration())
            use_frankenstein = false;

        double progress = std::min(1.0, elapsed / Config::MAX_TIME_SECONDS);
        double frank_cooldown;
        int frank_beam_width = Config::FRANKENSTEIN_BEAM_WIDTH;

        if (evaluator_->GetSolutionSize() <= Config::LARGE_INSTANCE_THRESHOLD) {

            frank_cooldown = 5.0;
        }
        else if (evaluator_->GetSolutionSize() > Config::HUGE_INSTANCE_THRESHOLD) {

            frank_cooldown = 45.0;
            frank_beam_width = 30;
        }
        else {

            frank_cooldown = 10.0 + (progress * 20.0);
        }

        if (elapsed < 60.0) {
            frank_cooldown *= 0.5;
        }
        auto now_frank = std::chrono::steady_clock::now();
        double seconds_since_last_frank = std::chrono::duration<double>(now_frank - last_greedy_assembly_time_).count();

        if (seconds_since_last_frank < frank_cooldown)
            use_frankenstein = false;

        if (use_frankenstein &&
            route_pool_.HasNewRoutesSince(last_routes_added_snapshot_)) {
            last_routes_added_snapshot_ = current_updates;
            auto t_beam_start = std::chrono::high_resolution_clock::now();
            Individual frankenstein = route_pool_.SolveBeamSearch(
                evaluator_, split_, frank_beam_width);
            auto t_beam_end = std::chrono::high_resolution_clock::now();
            if (Config::SHOW_LOGS) prof_frankenstein_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_beam_end - t_beam_start).count();
            if (frankenstein.IsEvaluated() && frankenstein.GetFitness() < 1e9) {
                int vnd_iters = Config::FRANKENSTEIN_VND_ITERS;
                if (elapsed > Config::MAX_TIME_SECONDS * 0.8)
                    vnd_iters = Config::FRANKENSTEIN_VND_ITERS_LATE;

                int passes = Config::FRANKENSTEIN_VND_PASSES;

                if (evaluator_->GetSolutionSize() > 2000 && elapsed > 60.0) {
                    vnd_iters = std::min(vnd_iters, vnd_max_adaptive_);
                    passes = 1;
                }

                bool improved = false;

                if (evaluator_->GetSolutionSize() > Config::HUGE_INSTANCE_THRESHOLD) {
                    auto t_vnd_start = std::chrono::high_resolution_clock::now();
                    improved = local_search_.RunHugeInstanceVND(frankenstein, 2);
                    auto t_vnd_end = std::chrono::high_resolution_clock::now();
                    if (Config::SHOW_LOGS) prof_frankenstein_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_vnd_end - t_vnd_start).count();
                }
                else {
                    for (int pass = 0; pass < passes; ++pass) {
                        auto t_vnd_start = std::chrono::high_resolution_clock::now();
                        if (local_search_.RunVND(frankenstein, vnd_iters, true, true, true, false, true))
                            improved = true;
                        else
                            break;
                        auto t_vnd_end = std::chrono::high_resolution_clock::now();
                        if (Config::SHOW_LOGS) prof_frankenstein_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_vnd_end - t_vnd_start).count();
                    }
                }

                if (improved) {
                    frankenstein.Canonicalize();
                    frankenstein.SetFitness(SafeEvaluate(frankenstein));
                }

                if (!ContainsSolution(frankenstein)) {
                    std::lock_guard<std::mutex> lock(population_mutex_);

                    bool force_injected = false;
                    std::uniform_real_distribution<double> d_force(0.0, 1.0);
                    if (d_force(rng_) < Config::FRANKENSTEIN_FORCE_INJECT_PROB) {
                        int victim_idx = rng_() % population_.size();
                        if (population_[victim_idx].GetFitness() >
                            current_best_.GetFitness() + 1e-6) {
                            population_[victim_idx] = frankenstein;
                            LogFrankensteinResult(frankenstein.GetFitness(), true, false);
                            force_injected = true;
                        }
                    }

                    if (!force_injected) {
                        int worst = GetWorstBiasedIndex();
                        if (worst >= 0) {
                            if (frankenstein.GetFitness() < population_[worst].GetFitness()) {
                                population_[worst] = frankenstein;
                                LogFrankensteinResult(frankenstein.GetFitness(), false, true);
                            }
                        }
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(best_mutex_);
                    if (frankenstein.GetFitness() < current_best_.GetFitness()) {
                        current_best_ = frankenstein;
                        stagnation_count_ = 0;
                    }
                }
            }
            last_greedy_assembly_time_ = now;
        }
    }
}

bool Island::ContainsSolution(const Individual& ind) const {
    uint64_t h = HashGenotype64(ind.GetGenotype());
    int num_groups = evaluator_->GetNumGroups();
    const auto& perm = evaluator_->GetPermutation();
    int genotype_size = static_cast<int>(ind.GetGenotype().size());

    int bpd_clone_threshold =
        std::max(10, genotype_size * 10 / 100);

    for (const auto& p : population_) {

        if (HashGenotype64(p.GetGenotype()) == h)
            return true;


        int bpd = const_cast<Island*>(this)->CalculateBrokenPairsDistance(
            ind, p, perm, num_groups);

        if (bpd < bpd_clone_threshold) {
            return true;
        }
    }
    return false;
}



int Island::ApplyMicroSplitMutation(Individual& child) {
    double stagnation_factor = std::min(1.0, (double)stagnation_count_ / 2000.0);


    int intensity;
    if (IsExploration()) {
        // id=0 -> level 0 (large), id=2 -> level 1 (medium), id=4 -> level 2 (small)
        intensity = (id_ / 2) % 3; // 0, 1, or 2
    }
    else {
        intensity = 2; // EXPLOIT uses only small windows
    }
    auto t_mut_start = std::chrono::high_resolution_clock::now();
    bool success = mutator_.ApplyMicroSplitMutation(child, stagnation_factor,
        intensity, rng_);
    auto t_mut_end = std::chrono::high_resolution_clock::now();
    if (Config::SHOW_LOGS) prof_mutation_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_mut_end - t_mut_start).count();
    (void)success; // suppress unused warning
    return 0;
}

int Island::ApplyMutation(Individual& child, bool is_endgame) {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    bool mutated = false;
    bool strong_mutation =
        false;


    if (IsExploration()) {
        double rnd = d(rng_);
        auto t_mut_start = std::chrono::high_resolution_clock::now();

        //again, specialized operators per island
        switch (id_) {
        case 0:

            if (rnd < 0.65) {

                double intensity = 0.3 + d(rng_) * 0.4;
                mutator_.ApplyRuinRecreate(child, intensity, false, rng_);
                strong_mutation = true;
            }
            else {

                mutator_.AggressiveMutate(child, rng_);
                strong_mutation = true;
            }
            mutated = true;
            break;

        case 2:
            if (rnd < 0.55) {

                if (mutator_.ApplyMergeSplit(child, rng_)) {
                    strong_mutation = true;
                }
            }
            else {

                int level = std::uniform_int_distribution<int>(0, 2)(rng_);
                double stagnation_factor = std::min(1.0, (double)stagnation_count_ / 2000.0);
                mutator_.ApplyMicroSplitMutation(child, stagnation_factor, level, rng_);
                strong_mutation = true;
            }
            mutated = true;
            break;

        case 4:

            if (rnd < 0.45) {

                mutator_.ApplySmartSpatialMove(child, rng_);
            }
            else if (rnd < 0.70) {

                mutator_.ApplyRuinRecreate(child, 0.15, false, rng_);
                strong_mutation = true;
            }
            else if (rnd < 0.75) {

                mutator_.AggressiveMutate(child, rng_);
                strong_mutation = true;
            }
            else {
                if (mutator_.EliminateReturns(child, rng_)) {
                    strong_mutation = true;
                }
            }
            mutated = true;
            break;

        default:
            //fallback
            if (rnd < 0.50) {
                double intensity = 0.3 + d(rng_) * 0.4;
                mutator_.ApplyRuinRecreate(child, intensity, false, rng_);
                strong_mutation = true;
            }
            else if (rnd < 0.75) {
                mutator_.ApplyMicroSplitMutation(child, 1.0, 1, rng_);
                strong_mutation = true;
            }
            else {
                mutator_.AggressiveMutate(child, rng_);
                strong_mutation = true;
            }
            mutated = true;
            break;
        }

        auto t_mut_end = std::chrono::high_resolution_clock::now();
        if (Config::SHOW_LOGS) prof_mutation_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_mut_end - t_mut_start).count();

        // rturn early for EXPLORE - exclusive operator already applied
        if (strong_mutation)
            return 2;
        if (mutated)
            return 1;
        return 0;
    }

    //EXPLOIT
    if (d(rng_) < p_microsplit_) {
        auto t_mut_start = std::chrono::high_resolution_clock::now();
        ApplyMicroSplitMutation(child);
        auto t_mut_end = std::chrono::high_resolution_clock::now();
        if (Config::SHOW_LOGS) prof_mutation_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_mut_end - t_mut_start).count();
        strong_mutation = true;
        mutated = true;
    }

    int problem_size = evaluator_->GetSolutionSize();
    if (IsExploitation() && problem_size > Config::LARGE_INSTANCE_THRESHOLD && d(rng_) < 0.02) {
        double current_cost = child.GetFitness();
        std::vector<int> guide = GetBestSolution();
        if (local_search_.TryPathRelinking(child.AccessGenotype(), current_cost, guide)) {
            mutated = true;
            strong_mutation = true;
        }
    }


    if (d(rng_) < Config::EXPLOITATION_MUTATION_PROB) {
        double rnd = d(rng_);
        constexpr double MUT_SPATIAL = 0.35;
        constexpr double MUT_AGGRESSIVE = 0.05;

        auto t_mut_start = std::chrono::high_resolution_clock::now();
        if (rnd < MUT_AGGRESSIVE) {
            mutator_.AggressiveMutate(child, rng_);
        }
        else if (rnd < MUT_SPATIAL) {
            mutator_.ApplySmartSpatialMove(child, rng_);
        }
        else {
            LNSStrategy strategy = aos_.SelectLNSStrategy(rng_);
            mutator_.ApplyAdaptiveLNS(child, strategy, (1 - current_structural_diversity_),
                true, rng_);
        }
        auto t_mut_end = std::chrono::high_resolution_clock::now();
        if (Config::SHOW_LOGS) prof_mutation_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_mut_end - t_mut_start).count();
        mutated = true;
    }

    if (d(rng_) < p_mergesplit_) {
        if (mutator_.ApplyMergeSplit(child, rng_)) {
            mutated = true;
            strong_mutation = true;
        }
    }

    if (d(rng_) < p_retminimizer_) {
        if (mutator_.ApplyReturnMinimizer(child, rng_)) {
            mutated = true;
        }
    }


    if (d(rng_) < p_retminimizer_ * 0.5) {
        bool is_tight = (1.0 * evaluator_->GetTotalDemand() / (evaluator_->GetNumGroups() * evaluator_->GetCapacity())) > 0.90;
        if (!is_tight && mutator_.EliminateReturns(child, rng_)) {
            mutated = true;
            strong_mutation = true;
        }
    }

    double swap_chance = is_endgame ? 0.50 : p_swap3_;
    if (d(rng_) < swap_chance) {
        if (local_search_.Try3Swap(child.AccessGenotype())) {
            mutated = true;
            strong_mutation = true;
        }
    }

    double current_p_swap4 = is_endgame ? 0.40 : p_swap4_;
    if (d(rng_) < current_p_swap4) {
        if (local_search_.Try4Swap(child.AccessGenotype())) {
            mutated = true;
            strong_mutation = true;
        }
    }

    if (strong_mutation)
        return 2;
    if (mutated)
        return 1;
    return 0;
}


static void ApplyDoubleBridge(std::vector<int>& perm, std::mt19937& rng) {
    int n = static_cast<int>(perm.size());
    if (n < 8) return;

    std::vector<int> cuts(4);
    cuts[0] = 1 + rng() % (n / 4);
    cuts[1] = cuts[0] + 1 + rng() % (n / 4);
    cuts[2] = cuts[1] + 1 + rng() % (n / 4);
    cuts[3] = std::min(n - 1, cuts[2] + 1 + (int)(rng() % (n / 4)));

    std::vector<int> result;
    result.reserve(n);

    for (int i = 0; i <= cuts[0]; ++i) result.push_back(perm[i]);
    for (int i = cuts[2] + 1; i <= cuts[3]; ++i) result.push_back(perm[i]);
    for (int i = cuts[1] + 1; i <= cuts[2]; ++i) result.push_back(perm[i]);
    for (int i = cuts[0] + 1; i <= cuts[1]; ++i) result.push_back(perm[i]);
    for (int i = cuts[3] + 1; i < n; ++i) result.push_back(perm[i]);

    perm = result;
}

void Island::Catastrophy() {
    std::vector<Individual> new_pop;
    new_pop.reserve(population_size_);
    const auto& original_perm = evaluator_->GetPermutation();


    double elite_pct = 0.20;
    if (catastrophes_since_improvement_ > 0) elite_pct = 0.10;
    if (catastrophes_since_improvement_ > 2) elite_pct = 0.05;
    if (catastrophes_since_improvement_ > 4) elite_pct = 0.00;

    int n_elite = std::max(1, (int)(population_size_ * elite_pct));
    if (catastrophes_since_improvement_ > 5) n_elite = 1;


    int remaining = population_size_ - n_elite;
    int n_shuffle = remaining * 0.20;
    int n_double_bridge = remaining * 0.20;
    int n_history = remaining * 0.50;
    int n_random = remaining - n_shuffle - n_double_bridge - n_history;

    if (IsExploitation() && catastrophes_since_improvement_ > 3 && Config::SHOW_LOGS) {
        std::cout << "\033[33m [ESCALATION I" << id_ << "] Reducing elite to " << n_elite
            << " (Failures: " << catastrophes_since_improvement_ << ")\033[0m\n";
    }


    int n_elite_perturbed = std::max(1, population_size_ / 3);
    if (catastrophes_since_improvement_ > 2) n_elite_perturbed = std::max(1, population_size_ / 5);

    {
        std::lock_guard<std::mutex> lock(population_mutex_);
        std::sort(population_.begin(), population_.end());

        if (!population_.empty()) {
            new_pop.push_back(population_[0]);
        }

        for (int i = 1; i < n_elite_perturbed && i < (int)population_.size(); ++i) {
            Individual elite_copy = population_[i];
            mutator_.ApplyRuinRecreate(elite_copy, 0.30, IsExploitation(), rng_);
            elite_copy.SetNative(true);
            elite_copy.SetHomeIsland(id_);
            double fit = SafeEvaluate(elite_copy);
            elite_copy.SetFitness(fit);
            new_pop.push_back(elite_copy);
        }
    }

    n_elite = new_pop.size();

    for (int i = 0; i < n_shuffle; ++i) {
        std::vector<int> shuffled_perm = original_perm;
        std::shuffle(shuffled_perm.begin(), shuffled_perm.end(), rng_);

        SplitResult result = split_.RunLinear(shuffled_perm);
        if (result.feasible && !result.group_assignment.empty()) {
            Individual indiv(result.group_assignment);
            indiv.SetNative(true);
            indiv.SetHomeIsland(id_);
            double fit = SafeEvaluate(indiv.GetGenotype());
            indiv.SetFitness(fit);
            new_pop.push_back(indiv);
        }
        else {
            // fallback
            Individual indiv(evaluator_->GetSolutionSize());
            InitIndividual(indiv, INITIALIZATION_TYPE::RANDOM);
            indiv.SetNative(true);
            indiv.SetHomeIsland(id_);
            indiv.SetFitness(SafeEvaluate(indiv.GetGenotype()));
            new_pop.push_back(indiv);
        }
    }

    for (int i = 0; i < n_double_bridge; ++i) {
        std::vector<int> perturbed_perm = original_perm;
        ApplyDoubleBridge(perturbed_perm, rng_);

        SplitResult result = split_.RunLinear(perturbed_perm);
        if (result.feasible && !result.group_assignment.empty()) {
            Individual indiv(result.group_assignment);
            indiv.SetNative(true);
            indiv.SetHomeIsland(id_);
            double fit = SafeEvaluate(indiv.GetGenotype());
            indiv.SetFitness(fit);
            new_pop.push_back(indiv);
        }
        else {

            Individual indiv(evaluator_->GetSolutionSize());
            InitIndividual(indiv, INITIALIZATION_TYPE::CHUNKED);
            indiv.SetNative(true);
            indiv.SetHomeIsland(id_);
            indiv.SetFitness(SafeEvaluate(indiv.GetGenotype()));
            new_pop.push_back(indiv);
        }
    }

    for (int i = 0; i < n_history; ++i) {
        Individual indiv(evaluator_->GetSolutionSize());
        InitIndividual(indiv, INITIALIZATION_TYPE::RANDOM);
        indiv.SetNative(true);
        indiv.SetHomeIsland(id_);

        double fit = EvaluateWithHistoryPenalty(indiv.GetGenotype());
        indiv.SetFitness(fit);
        new_pop.push_back(indiv);
    }

    for (int i = 0; i < n_random; ++i) {
        Individual indiv(evaluator_->GetSolutionSize());
        if (i % 2 == 0) {
            InitIndividual(indiv, INITIALIZATION_TYPE::RANDOM);
        }
        else {
            InitIndividual(indiv, INITIALIZATION_TYPE::BIN_PACKING);
        }
        indiv.SetNative(true);
        indiv.SetHomeIsland(id_);
        indiv.SetFitness(SafeEvaluate(indiv.GetGenotype()));
        new_pop.push_back(indiv);
    }

    {
        std::lock_guard<std::mutex> lock(population_mutex_);
        population_ = new_pop;
        UpdateBiasedFitness();
    }


    if (IsExploitation()) {
        adapt_swap_.success_rate = 0.5;
        adapt_ejection_.success_rate = 0.5;
        adapt_swap3_.success_rate = 0.5;
        adapt_swap4_.success_rate = 0.5;
    }

    aos_.Reset();


    stagnation_count_ = 0;

    //immunity after catastrophy
    immune_until_time_ =
        std::chrono::steady_clock::now() + std::chrono::seconds(15);
}

void Island::UpdateBiasedFitness() {
    int pop_size = static_cast<int>(population_.size());
    if (pop_size == 0)
        return;

    Individual best;
    {
        std::lock_guard<std::mutex> lock(best_mutex_);
        best = current_best_;
    }

    const std::vector<int>& perm = evaluator_->GetPermutation();
    int num_groups = evaluator_->GetNumGroups();

    double total_population_bpd = 0.0;
    int measurements_count = 0;


    int num_clients = evaluator_->GetSolutionSize();
    int sample_step = 1;
    if (num_clients > Config::HUGE_INSTANCE_THRESHOLD) {
        sample_step = std::max(1, pop_size / 10);
    }
    else if (num_clients > Config::LARGE_INSTANCE_THRESHOLD) {
        sample_step = std::max(1, pop_size / 20);
    }

    for (int i = 0; i < pop_size; ++i) {
        if (i % sample_step == 0) {
            int bpd =
                CalculateBrokenPairsDistance(population_[i], best, perm, num_groups);
            population_[i].SetDiversityScore(static_cast<double>(bpd));
            total_population_bpd += bpd;
            measurements_count++;
        }
        else {

            int prev_sampled = (i / sample_step) * sample_step;
            population_[i].SetDiversityScore(
                population_[prev_sampled].GetDiversityScore());
        }
    }

    if (measurements_count > 0) {
        double avg_raw_bpd = total_population_bpd / measurements_count;
        double raw_diversity = avg_raw_bpd / (double)evaluator_->GetSolutionSize();
        double range = max_diversity_baseline_ - min_diversity_baseline_;
        if (range > 0.001) {
            current_structural_diversity_ =
                (raw_diversity - min_diversity_baseline_) / range;
            current_structural_diversity_ =
                std::max(0.0, std::min(1.0, current_structural_diversity_));
        }
        else {
            current_structural_diversity_ = 0.5;
        }
    }
    else {
        current_structural_diversity_ = 0.0;
    }

    //sort by diversity
    std::vector<int> indices(pop_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return population_[a].GetDiversityScore() >
            population_[b].GetDiversityScore();
        });

    std::vector<int> rank_diversity(pop_size);
    for (int r = 0; r < pop_size; ++r)
        rank_diversity[indices[r]] = r;

    std::vector<int> rank_fitness(pop_size);
    std::vector<int> fit_indices(pop_size);
    std::iota(fit_indices.begin(), fit_indices.end(), 0);
    std::sort(fit_indices.begin(), fit_indices.end(), [&](int a, int b) {
        return population_[a].GetFitness() < population_[b].GetFitness();
        });
    for (int r = 0; r < pop_size; ++r)
        rank_fitness[fit_indices[r]] = r;

    double elite_ratio = 1.0 / (double)pop_size;
    for (int i = 0; i < pop_size; ++i) {
        double biased = (double)rank_fitness[i] + (1.0 - elite_ratio) * (double)rank_diversity[i];
        population_[i].SetBiasedFitness(biased);
    }
}
int Island::CalculateBrokenPairsDistance(const Individual& ind1, const Individual& ind2,
    const std::vector<int>& permutation,
    int num_groups) {
    const std::vector<int>& g1 = ind1.GetGenotype();
    const std::vector<int>& g2 = ind2.GetGenotype();
    int size = static_cast<int>(g1.size());
    if (size == 0)
        return 0;

    if (pred1.size() < size) pred1.resize(size);
    if (pred2.size() < size) pred2.resize(size);
    if (last_in_group1.size() < num_groups) last_in_group1.resize(num_groups, -1);
    if (last_in_group2.size() < num_groups) last_in_group2.resize(num_groups, -1);

    std::fill(last_in_group1.begin(), last_in_group1.end(), -1);
    std::fill(last_in_group2.begin(), last_in_group2.end(), -1);

    for (int customer_id : permutation) {
        int idx = customer_id - 2;
        if (idx < 0 || idx >= size)
            continue;

        int group1 = g1[idx];
        if (group1 >= 0 && group1 < num_groups) {
            pred1[idx] = last_in_group1[group1];
            last_in_group1[group1] = idx;
        }
        else {
            pred1[idx] = -2;
        }

        int group2 = g2[idx];
        if (group2 >= 0 && group2 < num_groups) {
            pred2[idx] = last_in_group2[group2];
            last_in_group2[group2] = idx;
        }
        else {
            pred2[idx] = -2;
        }
    }

    int distance = 0;
    for (int i = 0; i < size; ++i) {
        if (pred1[i] != pred2[i])
            distance++;
    }
    return distance;
}


int Island::CalculateSampledBPD(const Individual& ind1, const Individual& ind2,
    int sample_size) {
    const std::vector<int>& g1 = ind1.GetGenotype();
    const std::vector<int>& g2 = ind2.GetGenotype();
    int size = static_cast<int>(g1.size());

    if (size == 0) return 0;
    if (size <= sample_size) {

        return CalculateBrokenPairsDistance(ind1, ind2,
            evaluator_->GetPermutation(), evaluator_->GetNumGroups());
    }

    int mismatches = 0;
    std::uniform_int_distribution<int> dist(0, size - 1);

    for (int s = 0; s < sample_size; ++s) {
        int idx = dist(rng_);
        if (g1[idx] != g2[idx]) {
            mismatches++;
        }
    }

    return (mismatches * size) / sample_size;
}



Individual Island::CrossoverNeighborBased(const Individual& p1,
    const Individual& p2) {
    const std::vector<int>& g1 = p1.GetGenotype();
    const std::vector<int>& g2 = p2.GetGenotype();
    int size = static_cast<int>(g1.size());
    Individual child(size);
    std::vector<int>& child_genes = child.AccessGenotype();

    if (size == 0 || !geometry_.HasNeighbors()) {
        //fallback to uniform
        for (int i = 0; i < size; ++i) {
            child_genes[i] = (rng_() % 2 == 0) ? g1[i] : g2[i];
        }
        return child;
    }

    //pick a random center
    std::uniform_int_distribution<int> dist_idx(0, size - 1);
    int center_idx = dist_idx(rng_);

    // get center s neighbors - these come from p1
    const auto& neighbors = geometry_.GetNeighbors(center_idx);
    std::unordered_set<int> neighbor_set(neighbors.begin(), neighbors.end());
    neighbor_set.insert(center_idx); // include center itself

    for (int i = 0; i < size; ++i) {
        child_genes[i] = (neighbor_set.count(i) > 0) ? g1[i] : g2[i];
    }
    return child;
}



Individual Island::Crossover(const Individual& p1, const Individual& p2) {

    if (evaluator_->GetSolutionSize() > 800) {
        return CrossoverNeighborBased(p1, p2);
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng_);
    double sequence_prob;

    if (id_ <= 1)
        sequence_prob = 0.31;
    else if (id_ <= 3)
        sequence_prob = 0.4;
    else
        sequence_prob = 0.7;

    if (r < sequence_prob)
        return ApplySREX(p1, p2);
    else
        return CrossoverNeighborBased(p1, p2);
}



void Island::ApplySplitToIndividual(Individual& indiv) {
    const std::vector<int>& global_perm = evaluator_->GetPermutation();
    int fleet_limit = evaluator_->GetNumGroups();
    SplitResult result = split_.RunLinear(global_perm);

    if (result.feasible) {
        std::vector<int>& genes = indiv.AccessGenotype();
        if (result.group_assignment.size() != genes.size())
            return;

        int routes_count = static_cast<int>(result.optimized_routes.size());
        for (size_t i = 0; i < genes.size(); ++i) {
            int assigned_route_id = result.group_assignment[i];
            genes[i] = (assigned_route_id < fleet_limit)
                ? assigned_route_id
                : (assigned_route_id % fleet_limit);
        }

        int excess_vehicles =
            (routes_count > fleet_limit) ? (routes_count - fleet_limit) : 0;
        indiv.SetFitness(result.total_cost);
        indiv.SetReturnCount(excess_vehicles);
    }
    else {
        indiv.SetFitness(1.0e30);
    }
}


void Island::UpdateAdaptiveParameters() {
    double relative_div = 0.0;

    if (max_diversity_baseline_ > 1e-9) {
        relative_div = (current_structural_diversity_ - min_diversity_baseline_) /
            (max_diversity_baseline_ - min_diversity_baseline_);
    }
    relative_div = std::max(0.0, std::min(1.0, relative_div));


    double chaos = current_structural_diversity_;


    double base_mut_prob = IsExploration() ? Config::EXPLORATION_MUTATION_PROB
        : Config::EXPLOITATION_MUTATION_PROB;


    double dynamic_mut_prob = base_mut_prob +
        (Config::ADAPTIVE_CHAOS_BOOST * (1.0 - chaos)) -
        (Config::ADAPTIVE_CHAOS_PENALTY * chaos);


    double health = GetPopulationHealth();
    double health_boost = (1.0 - health) * 0.15;
    dynamic_mut_prob += health_boost;

    double mut_min, mut_max;
    if (IsExploration()) {
        switch (id_) {
        case 0:
            mut_min = Config::EXPLORE_I0_MUT_MIN;
            mut_max = Config::EXPLORE_I0_MUT_MAX;
            break;
        case 2:
            mut_min = Config::EXPLORE_I2_MUT_MIN;
            mut_max = Config::EXPLORE_I2_MUT_MAX;
            break;
        case 4:
            mut_min = Config::EXPLORE_I4_MUT_MIN;
            mut_max = Config::EXPLORE_I4_MUT_MAX;
            break;
        default:
            mut_min = Config::ADAPTIVE_MUT_MIN;
            mut_max = Config::ADAPTIVE_MUT_MAX;
            break;
        }
    }
    else {
        mut_min = Config::ADAPTIVE_MUT_MIN;
        mut_max = Config::ADAPTIVE_MUT_MAX;
    }
    dynamic_mut_prob = std::max(mut_min, std::min(mut_max, dynamic_mut_prob));

    p_microsplit_ = dynamic_mut_prob;
    p_retminimizer_ = dynamic_mut_prob * 0.6;
    p_mergesplit_ = dynamic_mut_prob * 0.5;

    if (IsExploitation()) {
        p_swap3_ = Config::EXPLOITATION_P_SWAP3;
        p_swap4_ = Config::EXPLOITATION_P_SWAP4;
        p_microsplit_ =
            std::max(Config::EXPLOITATION_MIN_MICROSPLIT, p_microsplit_);


        switch (id_) {
        case 1: p_mergesplit_ = Config::EXPLOIT_I1_OP_SPLIT_PROB; break;
        case 3: p_mergesplit_ = Config::EXPLOIT_I3_OP_SPLIT_PROB; break;
        case 5: p_mergesplit_ = Config::EXPLOIT_I5_OP_SPLIT_PROB; break;
        default: p_mergesplit_ = 0.05; break;
        }


        p_retminimizer_ = std::max(0.10, p_retminimizer_);

        if (chaos < 0.1) {
            p_retminimizer_ *= 1.5;
            p_mergesplit_ *= 1.5;
        }
    }
    else {

        p_swap3_ = 0.10;
        p_swap4_ = 0.00;
    }


    adaptive_mutation_rate_ = dynamic_mut_prob;
    adaptive_ruin_chance_ = dynamic_mut_prob;

    if (IsExploration()) {
        adaptive_vnd_prob_ = MapRange(relative_div, 0.0, 1.0, 0.10, 0.40);
    }
    else {
        adaptive_vnd_prob_ = MapRange(relative_div, 0.0, 1.0, 0.50, 0.95);
    }
}


void Island::ApplySuccessionAdaptive(std::vector<Individual>& offspring_pool) {

    std::lock_guard<std::mutex> lock(population_mutex_);

    if (!offspring_pool.empty()) {
        population_.reserve(population_.size() + offspring_pool.size());
        for (auto& child : offspring_pool) {
            population_.push_back(std::move(child));
        }
    }
    if (population_.empty())
        return;

    //deduplication
    std::sort(population_.begin(), population_.end(),
        [](const Individual& a, const Individual& b) {
            return a.GetFitness() < b.GetFitness();
        });

    std::vector<Individual> unique_candidates;
    unique_candidates.reserve(population_.size());
    std::unordered_set<uint64_t> used_hashes;

    for (const auto& ind : population_) {
        uint64_t h = HashGenotype64(ind.GetGenotype());
        if (used_hashes.find(h) == used_hashes.end()) {
            used_hashes.insert(h);
            unique_candidates.push_back(ind);
        }
    }
    population_ = std::move(unique_candidates);

    if ((int)population_.size() <= population_size_) {
        UpdateBiasedFitness();
        return;
    }

    // elite ratio varies based on diversity
    double relative_div = 0.0;
    if (max_diversity_baseline_ > 1e-9) {
        relative_div = (current_structural_diversity_ - min_diversity_baseline_) /
            (max_diversity_baseline_ - min_diversity_baseline_);
    }
    relative_div = std::max(0.0, std::min(1.0, relative_div));



    double elite_ratio;
    if (IsExploration()) {

        elite_ratio =
            MapRange(relative_div, 0.0, 1.0, Config::ELITE_RATIO_EXPLORATION_LOW,
                Config::ELITE_RATIO_EXPLORATION_HIGH);
    }
    else {

        elite_ratio =
            MapRange(relative_div, 0.0, 1.0, Config::ELITE_RATIO_EXPLOITATION_LOW,
                Config::ELITE_RATIO_EXPLOITATION_HIGH);
    }

    int elite_count = (int)(population_size_ * elite_ratio);
    elite_count = std::max(2, elite_count);

    std::vector<Individual> next_pop;
    next_pop.reserve(population_size_);
    std::unordered_set<int> taken_indices;


    for (int i = 0; i < (int)population_.size(); ++i) {
        if ((int)next_pop.size() >= elite_count)
            break;
        next_pop.push_back(population_[i]);
        taken_indices.insert(i);
    }

    if ((int)next_pop.size() < population_size_) {
        UpdateBiasedFitness();

        std::vector<int> biased_indices(population_.size());
        std::iota(biased_indices.begin(), biased_indices.end(), 0);
        std::sort(biased_indices.begin(), biased_indices.end(), [&](int a, int b) {
            return population_[a].GetBiasedFitness() <
                population_[b].GetBiasedFitness();
            });

        for (int idx : biased_indices) {
            if ((int)next_pop.size() >= population_size_)
                break;
            if (taken_indices.find(idx) == taken_indices.end()) {
                next_pop.push_back(population_[idx]);
                taken_indices.insert(idx);
            }
        }
    }

    //fallback
    if ((int)next_pop.size() < population_size_) {
        for (int i = 0; i < (int)population_.size(); ++i) {
            if ((int)next_pop.size() >= population_size_)
                break;
            if (taken_indices.find(i) == taken_indices.end()) {
                next_pop.push_back(population_[i]);
            }
        }
    }

    population_ = std::move(next_pop);


    std::sort(population_.begin(), population_.end(),
        [](const Individual& a, const Individual& b) {
            return a.GetFitness() < b.GetFitness();
        });

    UpdateBiasedFitness();
}


int Island::SelectParentIndex() {
    if (population_.empty())
        return -1;
    int pop_size = static_cast<int>(population_.size());
    std::uniform_int_distribution<int> dist(0, pop_size - 1);
    double best_val = std::numeric_limits<double>::max();
    int best_idx = -1;


    int tournament_size;
    if (IsExploration()) {
        tournament_size = (evaluator_->GetSolutionSize() <= Config::LARGE_INSTANCE_THRESHOLD)
            ? 3
            : Config::EXPLORATION_TOURNAMENT_SIZE;
    }
    else {
        tournament_size = Config::EXPLOITATION_TOURNAMENT_SIZE;
    }
    int t_size = std::min(tournament_size, pop_size);

    for (int i = 0; i < t_size; ++i) {
        int idx = dist(rng_);
        if (population_[idx].GetBiasedFitness() < best_val) {
            best_val = population_[idx].GetBiasedFitness(); // pick by biased fitness
            best_idx = idx;
        }
    }
    return best_idx;
}

int Island::GetWorstBiasedIndex() const {
    if (population_.empty())
        return -1;

    std::vector<int> best_genotype_copy;
    {
        std::lock_guard<std::mutex> lock(best_mutex_);
        best_genotype_copy = current_best_.GetGenotype();
    }

    int pop_size = static_cast<int>(population_.size());
    int worst = -1;
    double max_val = -1.0;
    for (int i = 0; i < pop_size; ++i) {
        if (population_[i].GetGenotype() == best_genotype_copy)
            continue;
        if (population_[i].GetBiasedFitness() > max_val) {
            max_val = population_[i].GetBiasedFitness();
            worst = i;
        }
    }
    return (worst == -1) ? GetWorstIndex() : worst;
}

int Island::GetWorstIndex() const {
    if (population_.empty())
        return -1;
    int pop_size = static_cast<int>(population_.size());
    int idx = 0;
    double worst = -1.0;
    for (int i = 0; i < pop_size; ++i) {
        if (population_[i].GetFitness() > worst) {
            worst = population_[i].GetFitness();
            idx = i;
        }
    }
    return idx;
}

void Island::InjectImmigrant(Individual& imigrant, bool force) {
    auto now = std::chrono::steady_clock::now();

    if (now < immune_until_time_) {
        double seconds_since_improve =
            std::chrono::duration<double>(now - last_improvement_time_).count();
        if (seconds_since_improve < 30.0) {
            return; //island is immune after catastrophy
        }
    }

    double best_fit;
    {
        std::lock_guard<std::mutex> lock(best_mutex_);
        best_fit = current_best_.GetFitness();
    }


    double fit;
    if (imigrant.IsEvaluated()) {
        fit = imigrant.GetFitness();
    }
    else {
        EvaluationResult res = evaluator_->EvaluateWithStats(imigrant.GetGenotype());
        fit = res.fitness;
        imigrant.SetFitness(fit);
        imigrant.SetReturnCount(res.returns);
    }

    if (fit == std::numeric_limits<double>::max())
        return;

    bool is_stuck = is_stuck_.load(std::memory_order_relaxed);


    double health = GetPopulationHealth();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();
    double time_ratio = elapsed / Config::MAX_TIME_SECONDS;

    double opportunity_threshold;
    if (health < Config::HEALTH_CRITICAL || time_ratio > 0.8) {
        opportunity_threshold = 0.999;
    }
    else if (health < Config::HEALTH_SICK || time_ratio > 0.5) {
        opportunity_threshold = 0.995;
    }
    else {
        opportunity_threshold = 0.99;
    }

    bool is_opportunity = (fit < best_fit * opportunity_threshold);

    if (!force && !is_stuck && !is_opportunity) {
        return;
    }

    std::lock_guard<std::mutex> lock(population_mutex_);

    // replace WORST BIASED fitness indiv
    int victim_idx = GetWorstBiasedIndex();
    if (victim_idx >= 0 && victim_idx < (int)population_.size()) {

        bool migrant_better = imigrant.GetFitness() < population_[victim_idx].GetFitness();
        bool victim_stagnant = population_[victim_idx].GetStagnation() > 200;


        int bpd_to_best;
        {
            Individual my_best_copy;
            {
                std::lock_guard<std::mutex> best_lock(best_mutex_);
                my_best_copy = current_best_;
            }
            bpd_to_best = CalculateBrokenPairsDistancePublic(imigrant, my_best_copy);
        }

        double diversity_pct = (evaluator_->GetSolutionSize() > Config::LARGE_INSTANCE_THRESHOLD)
            ? 0.08
            : 0.10;
        bool adds_diversity = bpd_to_best > (evaluator_->GetSolutionSize() * diversity_pct);

        if (migrant_better || victim_stagnant || adds_diversity) {
            population_[victim_idx] = imigrant;

            std::lock_guard<std::mutex> best_lock(best_mutex_);
            if (imigrant.GetFitness() < current_best_.GetFitness()) {
                current_best_ = imigrant;
                last_improvement_gen_ = current_generation_;
                last_improvement_time_ = now;
                catastrophes_since_improvement_ = 0;
            }
        }
    }

}





Individual Island::GetRandomEliteIndividual() {
    std::lock_guard<std::mutex> lock(population_mutex_);
    if (population_.empty())
        return current_best_;

    //pick random from top 30, pop is sorted
    int elite_size = std::max(1, (int)(population_.size() * 0.30));
    std::uniform_int_distribution<int> dist(0, elite_size - 1);
    return population_[dist(rng_)];
}



void Island::TryPullMigrant() {

    if (intensification_active_) {
        return;
    }


    if (!ring_predecessor_)
        return;

    auto now = std::chrono::steady_clock::now();


    if (now < migration_lock_until_) {
        return;
    }

    double seconds_since_migration =
        std::chrono::duration<double>(now - last_migration_time_).count();


    double seconds_since_catastrophe =
        std::chrono::duration<double>(now - last_catastrophy_time_).count();
    if (seconds_since_catastrophe < 6.7) {
        return;
    }


    double health = GetPopulationHealth();
    int problem_size = evaluator_->GetSolutionSize();


    double base_healthy = Config::MIGRATION_INTERVAL_HEALTHY;
    double base_sick = Config::MIGRATION_INTERVAL_SICK;
    if (problem_size <= 500) {
        base_healthy = 5.0;
        base_sick = 3.0;
    }
    else if (problem_size <= 1000) {
        base_healthy = 8.0;
        base_sick = 3.0;
    }


    double interval;
    if (health < Config::HEALTH_CRITICAL) {
        interval = base_sick;
    }
    else if (health < Config::HEALTH_SICK) {

        double ratio = (health - Config::HEALTH_CRITICAL) / (Config::HEALTH_SICK - Config::HEALTH_CRITICAL);
        interval = base_sick + ratio * (base_healthy - base_sick) * 0.5;
    }
    else {
        interval = base_healthy;
    }

    if (seconds_since_migration < interval)
        return;

    //ROUTE MIGRATION!
    {
        auto top_routes = ring_predecessor_->GetTopRoutes(5);
        if (!top_routes.empty()) {
            std::lock_guard<std::mutex> lock(best_mutex_);
            route_pool_.ImportRoutes(top_routes);
        }
    }


    std::uniform_real_distribution<double> dist_trickle(0.0, 1.0);
    double trickle_prob = (health < Config::HEALTH_SICK)
        ? Config::TRICKLE_PROB_SICK
        : Config::TRICKLE_PROB_HEALTHY;

    bool did_trickle = false;
    if (dist_trickle(rng_) < trickle_prob) {
        Individual migrant =
            ring_predecessor_->GetRandomEliteIndividual();
        double migrant_fit = migrant.GetFitness();
        double my_best_fit = current_best_.GetFitness();


        bool valid_fitness = (migrant_fit > 0 && migrant_fit < 1e9);

        bool acceptable_quality = (migrant_fit < my_best_fit * 1.30);

        if (valid_fitness && acceptable_quality) {
            {
                std::lock_guard<std::mutex> lock(population_mutex_);
                if (!ContainsSolution(migrant)) {
                    int victim = GetWorstBiasedIndex();
                    if (victim >= 0) {
                        population_[victim] = migrant;
                        did_trickle = true;
                        if (Config::SHOW_LOGS) {
                            std::cout << "\033[33m [MIG I" << id_ << "] Trickle Pull from I"
                                << ring_predecessor_->GetId() << " (Fit=" << (int)migrant_fit
                                << ", home=I" << migrant.GetHomeIsland() << ")\033[0m\n";
                        }
                    }
                }
            }
        }
    }


    if (!did_trickle) {

        Individual my_best;
        {
            std::lock_guard<std::mutex> lock(best_mutex_);
            my_best = current_best_;
        }

        Individual migrant = ring_predecessor_->GetRandomEliteIndividual();


        double bpd_threshold_pct = 0.1;
        int min_bpd_threshold = static_cast<int>(evaluator_->GetSolutionSize() * bpd_threshold_pct);

        int bpd_to_best = CalculateBrokenPairsDistancePublic(migrant, my_best);

        double migrant_fit = migrant.GetFitness();
        double my_best_fit = my_best.GetFitness();


        bool valid_fitness = (migrant_fit > 0 && migrant_fit < 1e9);

        bool acceptable_quality = (migrant_fit < my_best_fit * 1.30);

        if (bpd_to_best >= min_bpd_threshold && valid_fitness && acceptable_quality) {

            bool is_also_better = (migrant_fit < my_best_fit * 0.995);
            if (is_also_better && Config::SHOW_LOGS) {
                std::cout << "\033[32m [MIG I" << id_ << "] Diversity Pull (BETTER)! "
                    << (int)migrant_fit << " < " << (int)my_best_fit
                    << " (BPD=" << bpd_to_best << ")\033[0m" << std::endl;
            }
            if (Config::SHOW_LOGS) {
                std::cout << "\033[33m [MIG I" << id_ << "] Diversity Pull from I"
                    << ring_predecessor_->GetId() << " (Fit=" << (int)migrant_fit
                    << ", BPD=" << bpd_to_best << ", home=I" << migrant.GetHomeIsland() << ")\033[0m\n";
            }

            InjectImmigrant(migrant);
        }
    }

    last_migration_time_ = now;

}


void Island::CalibrateDiversity() {
    const int SAMPLE_SIZE = 100;
    int n = evaluator_->GetSolutionSize();
    int num_groups = evaluator_->GetNumGroups();
    const std::vector<int>& perm = evaluator_->GetPermutation();

    std::vector<Individual> random_samples;
    random_samples.reserve(SAMPLE_SIZE);

    for (int i = 0; i < SAMPLE_SIZE; ++i) {
        Individual rnd_ind(n);
        InitIndividual(rnd_ind, INITIALIZATION_TYPE::RANDOM);
        random_samples.push_back(std::move(rnd_ind));
    }

    double total_broken_pairs = 0.0;
    long long comparisons_count = 0;
    const int PROBES_PER_IND = 5;

    for (int i = 0; i < SAMPLE_SIZE; ++i) {
        for (int k = 0; k < PROBES_PER_IND; ++k) {
            int other_idx = rng_() % SAMPLE_SIZE;
            if (i == other_idx)
                continue;
            int dist = CalculateBrokenPairsDistance(
                random_samples[i], random_samples[other_idx], perm, num_groups);
            total_broken_pairs += dist;
            comparisons_count++;
        }
    }

    if (comparisons_count > 0) {
        double avg_dist = total_broken_pairs / (double)comparisons_count;
        max_diversity_baseline_ = avg_dist / (double)n;
    }
    else {
        max_diversity_baseline_ = 1.0;
    }
    CalibrateConvergence();
}

void Island::CalibrateConvergence() {
    const int SAMPLE_SIZE = 50;
    int n = evaluator_->GetSolutionSize();
    int num_groups = evaluator_->GetNumGroups();
    const std::vector<int>& perm = evaluator_->GetPermutation();

    Individual base_ind(n);
    InitIndividual(base_ind, INITIALIZATION_TYPE::CHUNKED);

    std::vector<Individual> converged_samples;
    converged_samples.reserve(SAMPLE_SIZE);
    converged_samples.push_back(base_ind);

    for (int i = 1; i < SAMPLE_SIZE; ++i) {
        Individual variant = base_ind;
        std::vector<int>& geno = variant.AccessGenotype();
        int num_mutations = 1 + (rng_() % 3);
        for (int m = 0; m < num_mutations; ++m) {
            int idx = rng_() % n;
            geno[idx] = rng_() % num_groups;
        }
        converged_samples.push_back(std::move(variant));
    }

    double total_broken_pairs = 0.0;
    long long comparisons_count = 0;
    const int PROBES_PER_IND = 5;

    for (int i = 0; i < SAMPLE_SIZE; ++i) {
        for (int k = 0; k < PROBES_PER_IND; ++k) {
            int other_idx = rng_() % SAMPLE_SIZE;
            if (i == other_idx)
                continue;
            int dist = CalculateBrokenPairsDistance(
                converged_samples[i], converged_samples[other_idx], perm, num_groups);
            total_broken_pairs += dist;
            comparisons_count++;
        }
    }

    if (comparisons_count > 0) {
        double avg_dist = total_broken_pairs / (double)comparisons_count;
        min_diversity_baseline_ = avg_dist / (double)n;
    }
    else {
        min_diversity_baseline_ = 0.0;
    }
}

double Island::MapRange(double value, double in_min, double in_max,
    double out_min, double out_max) const {
    if (in_max - in_min < 1e-9)
        return out_min;
    double clamped = std::max(in_min, std::min(in_max, value));
    return out_min + (out_max - out_min) * (clamped - in_min) / (in_max - in_min);
}

int Island::GetVndIterations() const {
    int base_min = IsExploration() ? Config::EXPLORATION_VND_MIN
        : Config::EXPLOITATION_VND_MIN;
    int base_max = IsExploration() ? Config::EXPLORATION_VND_MAX
        : Config::EXPLOITATION_VND_MAX;


    int problem_size = evaluator_->GetSolutionSize();
    if (problem_size > Config::HUGE_INSTANCE_THRESHOLD) {

        base_max = IsExploration() ? 1 : 8;
        base_min = 1;
    }
    else if (problem_size > Config::LARGE_INSTANCE_THRESHOLD) {

        base_max = IsExploration() ? 8 : 40;
        base_min = std::min(base_min, 3);
    }

    double health = GetPopulationHealth();

    double health_mult = 0.6 + 0.6 * health;

    double diversity_factor = current_structural_diversity_;
    double result = base_max - (diversity_factor * (base_max - base_min));

    // apply health multiplier
    result *= health_mult;

    return static_cast<int>(
        std::max((double)base_min, std::min((double)base_max, result)));
}



Individual Island::ApplySREX(const Individual& p1, const Individual& p2) {
    int num_clients = evaluator_->GetSolutionSize();
    const std::vector<int>& g1 = p1.GetGenotype();
    const std::vector<int>& g2 = p2.GetGenotype();

    if (g1.size() != num_clients || g2.size() != num_clients) {
        Individual rnd(num_clients);
        InitIndividual(rnd, INITIALIZATION_TYPE::RANDOM);
        return rnd;
    }

    std::vector<int> child_genotype(num_clients, -1);
    std::vector<bool> is_covered(num_clients, false);

    //lambda to build route map
    auto build_routes = [&](const std::vector<int>& g) {
        int max_g = 0;
        for (int x : g)
            if (x > max_g)
                max_g = x;
        if (max_g > num_clients)
            max_g = num_clients;
        std::vector<std::vector<int>> routes(max_g + 1);
        for (int i = 0; i < num_clients; ++i) {
            if (g[i] >= 0 && g[i] <= max_g)
                routes[g[i]].push_back(i);
        }
        return routes;
        };

    auto routes1 = build_routes(g1);
    auto routes2 = build_routes(g2);

    std::vector<int> active1, active2;
    for (size_t i = 0; i < routes1.size(); ++i)
        if (!routes1[i].empty())
            active1.push_back((int)i);
    for (size_t i = 0; i < routes2.size(); ++i)
        if (!routes2[i].empty())
            active2.push_back((int)i);

    if (!active1.empty())
        std::shuffle(active1.begin(), active1.end(), rng_);
    if (!active2.empty())
        std::shuffle(active2.begin(), active2.end(), rng_);

    int current_child_group = 0;
    std::vector<int> child_group_loads;
    child_group_loads.reserve(num_clients);

    // inherit 50% from P1
    int take1 = std::max(1, (int)active1.size() / 2);
    for (int i = 0; i < take1 && i < (int)active1.size(); ++i) {
        int g_idx = active1[i];
        int load = 0;
        for (int client : routes1[g_idx]) {
            child_genotype[client] = current_child_group;
            is_covered[client] = true;
            load += evaluator_->GetDemand(client + 2);
        }
        child_group_loads.push_back(load);
        current_child_group++;
    }

    // inherit from P2 if no conflict
    for (int g_idx : active2) {
        bool conflict = false;
        for (int client : routes2[g_idx]) {
            if (is_covered[client]) {
                conflict = true;
                break;
            }
        }
        if (!conflict) {
            int load = 0;
            for (int client : routes2[g_idx]) {
                child_genotype[client] = current_child_group;
                is_covered[client] = true;
                load += evaluator_->GetDemand(client + 2);
            }
            child_group_loads.push_back(load);
            current_child_group++;
        }
    }

    //use the regret 3 heuristic to assign unassigned clients
    std::vector<int> unassigned;
    for (int i = 0; i < num_clients; ++i)
        if (!is_covered[i])
            unassigned.push_back(i);




    while (!unassigned.empty()) {

        int best_client = -1;
        int best_group = -1;
        double max_regret = -1e30;

        for (int client : unassigned) {
            int demand = evaluator_->GetDemand(client + 2);


            double cost1 = 1e30, cost2 = 1e30,
                cost3 = 1e30;
            int group1 = -1;

            for (size_t g = 0; g < child_group_loads.size(); ++g) {
                if (child_group_loads[g] + demand <= capacity_) {
                    double load_ratio =
                        (double)(child_group_loads[g] + demand) / capacity_;
                    if (load_ratio < cost1) {
                        cost3 = cost2;
                        cost2 = cost1;
                        cost1 = load_ratio;
                        group1 = (int)g;
                    }
                    else if (load_ratio < cost2) {
                        cost3 = cost2;
                        cost2 = load_ratio;
                    }
                    else if (load_ratio < cost3) {
                        cost3 = load_ratio;
                    }
                }
            }

            //REGRET-3: sum of (2nd - 1st) + (3rd - 1st)
            double regret;
            if (group1 >= 0) {
                double r2 = (cost2 < 1e29) ? (cost2 - cost1) : 0.5;
                double r3 = (cost3 < 1e29) ? (cost3 - cost1) : 0.5;
                regret = r2 + r3;
                if (cost2 >= 1e29)
                    regret = 1e10;
            }
            else {
                regret = 1e20;
            }

            if (regret > max_regret) {
                max_regret = regret;
                best_client = client;
                best_group = group1;
            }
        }

        if (best_client < 0)
            break;

        int demand = evaluator_->GetDemand(best_client + 2);

        if (best_group >= 0) {

            child_genotype[best_client] = best_group;
            child_group_loads[best_group] += demand;
        }
        else {

            int forced_group = 0;
            if (!child_group_loads.empty()) {

                std::uniform_int_distribution<int> dist(0, (int)child_group_loads.size() - 1);
                forced_group = dist(rng_);
                //pick a random group and hopefully vnd will fix it
            }

            child_genotype[best_client] = forced_group;
            child_group_loads[forced_group] += demand;
        }

        unassigned.erase(
            std::remove(unassigned.begin(), unassigned.end(), best_client),
            unassigned.end());
    }

    Individual child(child_genotype);
    child.Canonicalize();
    return child;
}



std::vector<CachedRoute> Island::GetTopRoutes(int n) const {

    std::unique_lock<std::mutex> lock(best_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        return {};
    }
    return route_pool_.GetBestRoutes(n);
}

void Island::ProcessBroadcastBuffer() {
    std::vector<Individual> processing_queue;
    {
        std::lock_guard<std::mutex> lock(broadcast_mutex_);
        if (broadcast_buffer_.empty())
            return;
        processing_queue = std::move(broadcast_buffer_);
        broadcast_buffer_.clear();
    }


    if (processing_queue.size() > 3) {

        std::partial_sort(processing_queue.begin(), processing_queue.begin() + 3, processing_queue.end(),
            [](const Individual& a, const Individual& b) {
                return a.GetFitness() < b.GetFitness();
            });

        processing_queue.resize(3); // kep top only top 3
    }


    double my_fitness = GetBestFitness();

    for (auto& candidate : processing_queue) {
        double cand_fit = candidate.GetFitness();

        //skip if worse than 5% of my best
        if (cand_fit > my_fitness * 1.05)
            continue;


        int home = candidate.GetHomeIsland();
        bool from_explore = (home == 0 || home == 2 || home == 4);

        //exploit injects for pure diversity
        if (from_explore && IsExploitation()) {

            if (!ContainsSolution(candidate) && cand_fit < my_fitness * 1.05) {
                InjectImmigrant(candidate, true);

                if (cand_fit < my_fitness * 0.99 && Config::SHOW_LOGS) {
                    std::cout << "\033[36m [I" << id_
                        << " EXPLOIT] Injected diversity from I" << home
                        << " (fit=" << std::fixed << std::setprecision(0)
                        << cand_fit << ")\033[0m\n";
                }
            }
        }
        else {

            if (cand_fit >= my_fitness)
                continue;

            double fitness_gap = (my_fitness - cand_fit) / my_fitness;
            bool significantly_better = (fitness_gap > 0.01);

            bool different_enough = false;
            if (!significantly_better) {
                int bpd = CalculateBrokenPairsDistancePublic(current_best_, candidate);


                double threshold_pct;
                if (evaluator_->GetSolutionSize() > 2000) {
                    threshold_pct = 0.02;
                }
                else if (evaluator_->GetSolutionSize() > Config::LARGE_INSTANCE_THRESHOLD) {
                    threshold_pct = 0.08;
                }
                else {
                    threshold_pct = 0.10;
                }
                int threshold = static_cast<int>(evaluator_->GetSolutionSize() * threshold_pct);

                if (bpd > threshold) {
                    different_enough = true;
                }
            }

            if (significantly_better || different_enough) {
                if (significantly_better && fitness_gap > 0.02 && Config::SHOW_LOGS) {
                    std::cout << "\033[95m [BROADCAST I" << id_
                        << "] Accepting SUPERIOR broadcast from I"
                        << candidate.GetHomeIsland() << " (Gap: " << std::fixed
                        << std::setprecision(2) << (fitness_gap * 100.0)
                        << "%)\033[0m\n";
                }
                InjectImmigrant(
                    candidate, true);
            }
        }
    }
}

void Island::ReceiveBroadcastBest(const Individual& best) {

    std::lock_guard<std::mutex> lock(broadcast_mutex_);


    Individual imported = best;
    imported.SetNative(false);


    broadcast_buffer_.push_back(std::move(imported));
}

void Island::UpdateAdaptiveProbabilities() {

    //adaptive chance based on success rate

    auto update_rate = [this](AdaptiveOperator& op) {
        if (op.calls > 0) {
            double current_rate = static_cast<double>(op.wins) / op.calls;

            op.success_rate =
                ADAPT_ALPHA * current_rate + (1.0 - ADAPT_ALPHA) * op.success_rate;

            op.calls = 0;
            op.wins = 0;
        }
        };

    update_rate(adapt_swap_);
    update_rate(adapt_ejection_);
    update_rate(adapt_swap3_);
    update_rate(adapt_swap4_);
}

int Island::SelectAdaptiveOperator() {

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng_);

    if (r < ADAPT_EPSILON) {

        std::uniform_int_distribution<int> op_dist(0, 3);
        return op_dist(rng_);
    }
    else {

        double rates[4] = { adapt_swap_.success_rate, adapt_ejection_.success_rate,
                           adapt_swap3_.success_rate, adapt_swap4_.success_rate };
        int best = 0;
        for (int i = 1; i < 4; i++) {
            if (rates[i] > rates[best])
                best = i;
        }
        return best;
    }
}



void Island::TrackCacheResult(bool was_hit) {
    cache_result_window_.push_back(was_hit);
    if (was_hit)
        cache_hits_in_window_++;

    if (static_cast<int>(cache_result_window_.size()) > CACHE_WINDOW_SIZE) {
        if (cache_result_window_.front())
            cache_hits_in_window_--;
        cache_result_window_.pop_front();
    }
}

double Island::GetRecentCacheHitRate() const {
    if (cache_result_window_.empty())
        return 0.0;
    return static_cast<double>(cache_hits_in_window_) /
        cache_result_window_.size();
}

void Island::OnConvergenceWarning() {
    // 85-90% cache hit rate: Double mutation rate for EXPLORE islands
    if (IsExploration()) {
        convergence_mutation_boost_ = 2.0;
        if (Config::SHOW_LOGS) {
            std::cout << "\033[33m [CONV-WARN I" << id_
                << "] Cache hit >85% - Boosting mutation 2x\033[0m\n";
        }
    }
}

void Island::OnConvergenceAlarm() {
    // 90-95% cache hit rate: Force broadcast of best to all siblings
    convergence_alarm_active_ = true;

    if (!exploit_siblings_.empty()) {
        Individual best_copy;
        {
            std::lock_guard<std::mutex> lock(best_mutex_);
            best_copy = current_best_;
        }

        for (Island* sibling : exploit_siblings_) {
            if (sibling != nullptr) {
                sibling->ReceiveBroadcastBest(best_copy);
            }
        }
        if (Config::SHOW_LOGS) {
            std::cout << "\033[35m [CONV-ALARM I" << id_
                << "] Cache hit >90% - Force broadcasted best to "
                << exploit_siblings_.size() << " siblings\033[0m\n";
        }
    }

    // also increase mutation for exploration islands even more
    if (IsExploration()) {
        convergence_mutation_boost_ = 3.0;
    }
}

void Island::OnConvergenceCritical() {
    // >95% cache hit rate: Mini-catastrophy for EXPLOIT
    if (Config::SHOW_LOGS) {
        std::cout << "\033[91m [CONV-CRITICAL I" << id_
            << "] Cache hit >95% - ";
    }

    if (IsExploitation()) {

        std::lock_guard<std::mutex> lock(population_mutex_);
        int restart_count = population_.size() / 2;

        for (int i = 0; i < restart_count; ++i) {
            int victim_idx = rng_() % population_.size();

            if (population_[victim_idx].GetFitness() > current_best_.GetFitness() + 1e-6) {
                Individual new_ind(evaluator_->GetSolutionSize());
                InitIndividual(new_ind, INITIALIZATION_TYPE::RANDOM);
                double fit = SafeEvaluate(new_ind);
                new_ind.SetFitness(fit);
                population_[victim_idx] = new_ind;
            }
        }
        if (Config::SHOW_LOGS) {
            std::cout << "Restarted " << restart_count << " individuals\033[0m\n";
        }
    }
    else {

        convergence_mutation_boost_ = 5.0;
        if (Config::SHOW_LOGS) {
            std::cout << "Boosting mutation 5x\033[0m\n";
        }
    }


    cache_result_window_.clear();
    cache_hits_in_window_ = 0;
    convergence_alarm_active_ = false;
}

double Island::GetCacheMissRate() const {
    long long total = cache_hits_ + cache_misses_;
    if (total < 100) return 0.5; // brak wystarczających danych
    return static_cast<double>(cache_misses_) / total;
}

double Island::GetPopulationHealth() const {
    // Health Score: 0.0 = martwa populacja, 1.0 = zdrowa populacja
    // 4 komponenty dla lepszego wykrywania stagnacji

    auto now = std::chrono::steady_clock::now();

    // Komponent 1: Structural Diversity (25% wagi)
    double diversity_score = std::max(0.0, std::min(1.0, current_structural_diversity_));

    // Komponent 2: VND Success Rate (20% wagi)
    double vnd_success = 0.5; // default
    if (diag_vnd_calls_ > 50) {
        vnd_success = static_cast<double>(diag_vnd_improvements_) / diag_vnd_calls_;
    }

    // Komponent 3: Cache Miss Rate (20% wagi)
    double cache_miss = const_cast<Island*>(this)->GetCacheMissRate();

    // Komponent 4: Recent Improvement Score (35% wagi) - KLUCZOWY!
    // Spada do 0 gdy nie ma poprawy przez 5 minut
    double seconds_since_improve =
        std::chrono::duration<double>(now - last_improvement_time_).count();
    double improvement_decay_time = 300.0; // 5 minut
    double improvement_score = std::max(0.0, 1.0 - seconds_since_improve / improvement_decay_time);

    // Weighted average (suma = 100%)
    double health = 0.25 * diversity_score +
        0.20 * vnd_success +
        0.20 * cache_miss +
        0.35 * improvement_score;  // Największa waga!

    return std::max(0.0, std::min(1.0, health));
}

void Island::PerformCrossIslandPathRelinking() {
    if (!ring_predecessor_) return;

    auto now = std::chrono::steady_clock::now();
    double seconds_since_pr =
        std::chrono::duration<double>(now - last_cross_island_pr_time_).count();

    double health = GetPopulationHealth();

    // Interwał zależny od zdrowia: chora populacja = częstsze PR
    double pr_interval = (health < Config::HEALTH_SICK) ? 15.0 : 45.0;

    if (seconds_since_pr < pr_interval) return;

    // Pobierz najlepszego od poprzednika
    Individual neighbor_best = ring_predecessor_->GetBestIndividual();
    Individual my_best;
    {
        std::lock_guard<std::mutex> lock(best_mutex_);
        my_best = current_best_;
    }

    // Sprawdź BPD - czy warto robić PR?
    int bpd = CalculateBrokenPairsDistancePublic(my_best, neighbor_best);
    int min_bpd = static_cast<int>(evaluator_->GetSolutionSize() * 0.05);
    if (bpd < min_bpd) {
        last_cross_island_pr_time_ = now;
        return; // Za podobne - nie warto
    }

    // Ustaw guide i uruchom VND z PR
    local_search_.SetGuideSolution(neighbor_best.GetGenotype());
    Individual pr_candidate = my_best;
    auto t_vnd_start = std::chrono::high_resolution_clock::now();
    local_search_.RunVND(pr_candidate, 50, true, true, true, true, false);
    auto t_vnd_end = std::chrono::high_resolution_clock::now();
    prof_vnd_time_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t_vnd_end - t_vnd_start).count();

    // Oceń wynik
    double fit = SafeEvaluate(pr_candidate);
    pr_candidate.SetFitness(fit);

    std::lock_guard<std::mutex> lock(best_mutex_);
    if (fit < current_best_.GetFitness()) {
        if (Config::SHOW_LOGS) {
            std::cout << "\033[92m [CROSS-PR I" << id_ << "] SUCCESS! "
                << (int)current_best_.GetFitness() << " -> " << (int)fit
                << " (BPD=" << bpd << ", health=" << std::fixed
                << std::setprecision(2) << health << ")\033[0m\n";
        }
        current_best_ = pr_candidate;
        last_improvement_gen_ = current_generation_;
        last_improvement_time_ = now;
        catastrophes_since_improvement_ = 0; // RESET
    }

    last_cross_island_pr_time_ = now;
}

void Island::NuclearCatastrophe() {

    if (Config::SHOW_LOGS) {
        std::cout << "[DEBUG] I" << id_ << " Nuclear: Starting..." << std::endl;
    }


    stagnation_count_ = 0;
    catastrophes_since_improvement_ = 0;


    if (Config::SHOW_LOGS) {
        std::cout << "[DEBUG] I" << id_ << " Nuclear: Clearing Cache..." << std::endl;
    }
    local_cache_.Clear();


    {
        std::lock_guard<std::mutex> lock(best_mutex_);

    }


    LogNuclearCatastrophe(true);


    {
        if (Config::SHOW_LOGS) {
            std::cout << "[DEBUG] I" << id_ << " Nuclear: Wiping Population..." << std::endl;
        }
        std::lock_guard<std::mutex> lock(population_mutex_);
        population_.clear();
        population_.reserve(population_size_);

        int sol_size = evaluator_->GetSolutionSize();
        if (sol_size <= 0) sol_size = 100; //fallback
        if (Config::SHOW_LOGS) {
            std::cout << "[DEBUG] I" << id_ << " Nuclear: SolSize=" << sol_size << " PopSize=" << population_size_ << std::endl;
        }

        //total start from scratch by double-bridge perturbations of original perm
        const auto& original_perm = evaluator_->GetPermutation();

        for (int i = 0; i < population_size_; ++i) {
            std::vector<int> perturbed_perm = original_perm;
            ApplyDoubleBridge(perturbed_perm, rng_);

            SplitResult result = split_.RunLinear(perturbed_perm);
            if (result.feasible && !result.group_assignment.empty()) {
                Individual ind(result.group_assignment);
                double fit = SafeEvaluate(ind.GetGenotype());
                ind.SetFitness(fit);
                ind.SetNative(true);
                ind.SetHomeIsland(id_);
                population_.push_back(std::move(ind));
            }
            else {
                Individual ind(sol_size);
                InitIndividual(ind, INITIALIZATION_TYPE::RANDOM);
                double fit = SafeEvaluate(ind.GetGenotype());
                ind.SetFitness(fit);
                ind.SetNative(true);
                ind.SetHomeIsland(id_);
                population_.push_back(std::move(ind));
            }
        }
        if (Config::SHOW_LOGS) {
            std::cout << "[DEBUG] I" << id_ << " Nuclear: Population Refilled." << std::endl;
        }


    }

    auto now = std::chrono::steady_clock::now();
    migration_lock_until_ = now + std::chrono::seconds(60);

    current_structural_diversity_ = 1.0;
    LogNuclearCatastrophe(false);
}


void Island::LogStatus(long long time_since) {
    if (Config::SHOW_LOGS && current_generation_ % 100 == 0) {
        std::string mode = IsExploration() ? "EXPLORE" : "EXPLOIT";
        std::cout << " [I" << id_ << "/" << mode << "] Gen: " << current_generation_
            << " Best: " << current_best_.GetFitness()
            << " Stall: " << time_since
            << " PopHealth: " << std::fixed << std::setprecision(2) << GetPopulationHealth()
            << std::endl;
    }
}

void Island::LogCatastropheTrigger(const char* reason, double unique_ratio, double health, double vnd_rate) {
    if (Config::SHOW_LOGS) {
        std::cout << "\033[96m [CATASTROPHE I" << id_ << "] Trigger: " << reason
            << " (unique=" << std::fixed << std::setprecision(0)
            << (unique_ratio * 100) << "%, health="
            << std::setprecision(2) << health << ", VND="
            << std::setprecision(1) << vnd_rate << "%, fail_count=" << catastrophes_since_improvement_ << ")\033[0m\n";
    }
}

void Island::LogFrankensteinResult(double fitness, bool force_injected, bool improved_worst) {
    if (Config::SHOW_LOGS) {
        if (force_injected) {
            std::cout << "\033[35m [BEAM] [Island " << id_
                << "] Frankenstein FORCIBLY injected (Fit: "
                << fitness << ")\033[0m" << std::endl;
        }
        else if (improved_worst) {
            std::cout << "\033[35m [BEAM] [Island " << id_
                << "] Frankenstein injected into population (Fit: "
                << fitness << ")\033[0m"
                << std::endl;
        }
    }
}

void Island::LogNuclearCatastrophe(bool start) {
    if (Config::SHOW_LOGS) {
        if (start) {
            std::cout << "\033[1;35m [NUCLEAR CATASTROPHE I" << id_ << "] 5 Failed Catastrophes! EXILING & REBOOTING POPULATION. Migration Lock: 60s.\033[0m" << std::endl;
        }
        else {
            std::cout << "[DEBUG] I" << id_ << " Nuclear: Finished." << std::endl;
        }
    }
}

void Island::LogBroadcast(int recipient_id, double fitness) {
    if (Config::SHOW_LOGS) {
        std::cout << "\033[95m [BROADCAST I" << id_
            << " -> I" << recipient_id << "] New Global Best: "
            << fitness << "\033[0m" << std::endl;
    }
}

void Island::LogDiagnostics(const std::chrono::steady_clock::time_point& now_diag) {
    //advanced diagnostics
    if (!Config::SHOW_LOGS) {
        return;
    }

    std::unordered_set<uint64_t> unique_hashes;
    double best_pop_fit = 1e18, worst_pop_fit = 0;
    double sum_fit = 0;
    {
        std::lock_guard<std::mutex> lock(population_mutex_);
        for (const auto& ind : population_) {
            uint64_t h = HashGenotype64(ind.GetGenotype());
            unique_hashes.insert(h);
            double f = ind.GetFitness();
            sum_fit += f;
            if (f < best_pop_fit)
                best_pop_fit = f;
            if (f > worst_pop_fit)
                worst_pop_fit = f;
        }
    }


    int zr_count = 0;
    {
        std::lock_guard<std::mutex> lock(population_mutex_);
        for (const auto& ind : population_) {
            if (ind.GetReturnCount() == 0)
                zr_count++;
        }
    }

    int unique_count = (int)unique_hashes.size();
    double unique_pct = 100.0 * unique_count / population_size_;
    double avg_fit = sum_fit / population_size_;


    long long gens_since_improve = current_generation_ - last_improvement_gen_;

    double global_best;
    {
        std::lock_guard<std::mutex> lock(best_mutex_);
        global_best = current_best_.GetFitness();
    }
    double gap_to_best = best_pop_fit - global_best;

    double vnd_success_rate =
        (diag_vnd_calls_ > 0)
        ? (100.0 * diag_vnd_improvements_ / diag_vnd_calls_)
        : 0.0;

    double better_pct =
        (diag_offspring_total_ > 0)
        ? (100.0 * diag_offspring_better_ / diag_offspring_total_)
        : 0.0;


    std::string issues = "";
    if (unique_pct < 30.0)
        issues += "[CLONE_FLOOD] ";
    if (vnd_success_rate < 5.0 && diag_vnd_calls_ > 50)
        issues += "[VND_STUCK] ";
    if (gens_since_improve > 500 && IsExploitation())
        issues += "[STAGNANT] ";
    if ((worst_pop_fit - best_pop_fit) < 100 && IsExploration())
        issues += "[LOW_DIVERSITY] ";
    if (better_pct < 10.0 && diag_offspring_total_ > 50)
        issues += "[POOR_OFFSPRING] ";
    if (gap_to_best > 5000 && IsExploitation())
        issues += "[LAGGING] ";


    double srex_rate = (diag_srex_calls_ > 0)
        ? (100.0 * diag_srex_wins_ / diag_srex_calls_)
        : 0.0;
    double neighbor_rate =
        (diag_neighbor_calls_ > 0)
        ? (100.0 * diag_neighbor_wins_ / diag_neighbor_calls_)
        : 0.0;
    double pr_rate =
        (diag_pr_calls_ > 0) ? (100.0 * diag_pr_wins_ / diag_pr_calls_) : 0.0;


    std::ostringstream oss;
    oss << " [DIAG I" << id_ << " " << (IsExploration() ? "EXP" : "EXT") << "] "
        << "VND: " << diag_vnd_improvements_ << "/" << diag_vnd_calls_ << " ("
        << std::fixed << std::setprecision(0) << vnd_success_rate << "%) | ";

    if (IsExploration()) {
        oss << "XO: SREX " << diag_srex_wins_ << "/" << diag_srex_calls_ << "("
            << srex_rate << "%) "
            << "NBR " << diag_neighbor_wins_ << "/" << diag_neighbor_calls_ << "("
            << neighbor_rate << "%) | ";
    }
    else {

        oss << "XO:PR " << diag_pr_wins_ << "/" << diag_pr_calls_ << "("
            << pr_rate << "%) "
            << "| EpsG: S2=" << std::setprecision(0)
            << (adapt_swap_.success_rate * 100) << "% "
            << "Ej=" << (adapt_ejection_.success_rate * 100) << "% "
            << "S3=" << (adapt_swap3_.success_rate * 100) << "% "
            << "S4=" << (adapt_swap4_.success_rate * 100) << "% | ";
    }

    oss << "Uniq: " << unique_count << "/" << population_size_ << " ("
        << std::setprecision(0) << unique_pct << "%) | "
        << "Div: " << std::fixed << std::setprecision(3)
        << current_structural_diversity_ << " | "
        << "Gap: " << std::setprecision(0) << gap_to_best << " | "
        << "Stag: " << gens_since_improve << "g | "
        << "ZR: " << zr_count << " ";

    if (!issues.empty()) {
        oss << "\033[31m" << issues << "\033[0m";
    }
    oss << "\n";

    if (prof_generations_ > 0) {
        double avg_mut = (double)prof_mutation_time_us_ / prof_generations_ / 1000.0;
        double avg_vnd = (double)prof_vnd_time_us_ / prof_generations_ / 1000.0;
        double avg_eval = (double)prof_eval_time_us_ / prof_generations_ / 1000.0;
        double avg_total = (double)prof_total_time_us_ / prof_generations_ / 1000.0;
        double avg_bdc = (double)prof_broadcast_time_us_ / prof_generations_ / 1000.0;
        double avg_succ = (double)prof_succession_time_us_ / prof_generations_ / 1000.0;


        double avg_rpool = (double)prof_routepool_time_us_ / prof_generations_ / 1000.0;
        double avg_frank = (double)prof_frankenstein_time_us_ / prof_generations_ / 1000.0;
        double avg_stag = (double)prof_stagnation_time_us_ / prof_generations_ / 1000.0;
        double avg_div = (double)prof_diversity_time_us_ / prof_generations_ / 1000.0;
        double avg_offs = (double)prof_offspring_time_us_ / prof_generations_ / 1000.0;
        double accounted = avg_mut + avg_vnd + avg_eval + avg_bdc + avg_succ + avg_rpool + avg_frank + avg_stag + avg_div + avg_offs;
        double avg_other = avg_total - accounted;

        oss << " [PROF] Avg/Gen: Total=" << std::fixed << std::setprecision(2) << avg_total
            << "ms (VND=" << avg_vnd << " RP=" << avg_rpool << " Frank=" << avg_frank
            << " Stag=" << avg_stag << " Div=" << avg_div << " Offs=" << avg_offs
            << " Bdc=" << avg_bdc << " Succ=" << avg_succ << " Oth=" << avg_other << ")\n";
    }


    std::cout << oss.str();
}
