#include "LocalSearch.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>

using namespace LcVRPContest;

LocalSearch::LocalSearch(ThreadSafeEvaluator* evaluator,
    const ProblemGeometry* geometry, int id)
    : evaluator_(evaluator), geometry_(geometry), id_(id) {

    rng_.seed(static_cast<unsigned int>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count() +
        id * 6777));

    int n = evaluator_->GetSolutionSize();
    int g = evaluator_->GetNumGroups();

    //to avoid allocation
    vnd_routes_.resize(g);
    for (auto& r : vnd_routes_) {
        r.reserve(n * 2 / g + 16);//just to be sure
    }
    vnd_loads_.resize(g);

    client_indices_.resize(n);
    candidate_groups_.reserve(Config::NUM_NEIGHBORS + 5);


    dlb_.resize(n, false);

    InitializeRanks();


    if (evaluator_->HasMatrix()) {
        fast_matrix_ = evaluator_->GetFastDistanceMatrix();
        matrix_dim_ = evaluator_->GetDimension();
    }
}

void LocalSearch::InitializeRanks() {
    int dim = evaluator_->GetDimension();
    customer_ranks_.resize(dim + 1, 0);
    const auto& perm = evaluator_->GetPermutation();
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] >= 0 && perm[i] < (int)customer_ranks_.size()) {
            customer_ranks_[perm[i]] = static_cast<int>(i);
        }
    }
}


double LocalSearch::SimulateRouteCost(const std::vector<int>& route_nodes) const {
    if (route_nodes.empty())
        return 0.0;

    return evaluator_->GetRouteCost(route_nodes);
}

void LocalSearch::ResetDLB() { std::fill(dlb_.begin(), dlb_.end(), false); }


bool LocalSearch::OptimizeActiveSet(Individual& ind, int max_iter, bool allow_swap, bool allow_3swap, bool allow_ejection, bool allow_4swap, bool unlimited_moves) {
    std::vector<int>& genotype = ind.AccessGenotype();
    int num_groups = evaluator_->GetNumGroups();
    int num_clients = static_cast<int>(genotype.size());
    const double EPSILON = 1e-4;

    if (dlb_.size() != genotype.size()) {
        dlb_.assign(num_clients, false);
    }
    else {

        ResetDLB();
    }

    bool improvement = true;
    bool any_change = false;
    int iter = 0;


    int total_moves_checked = 0;
    const int LARGE_INSTANCE_MOVE_LIMIT = 15000;
    const bool is_large_instance = (num_clients > Config::LARGE_INSTANCE_THRESHOLD);
    const bool apply_move_limit = is_large_instance && !unlimited_moves;

    temp_route_buffer_.reserve(num_clients / num_groups + 50);


    const auto vnd_start_time = std::chrono::high_resolution_clock::now();
    const int VND_TIME_LIMIT_MS = 150;
    const bool apply_time_limit = (num_clients > 2000) && !unlimited_moves;
    int time_check_counter = 0;
    const int TIME_CHECK_INTERVAL = 100;

    while (improvement && iter < max_iter) {
        improvement = false;
        iter++;


        bool time_limit_exceeded = false;
        bool move_limit_exceeded = false;

        if (apply_move_limit && total_moves_checked > LARGE_INSTANCE_MOVE_LIMIT) {
            move_limit_exceeded = true;
            improvement = false;
        }

        for (int client_idx : client_indices_) {
            if (move_limit_exceeded) continue;
            if (time_limit_exceeded) continue;

            if (apply_time_limit && (++time_check_counter % TIME_CHECK_INTERVAL == 0)) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - vnd_start_time).count();
                if (elapsed_ms > VND_TIME_LIMIT_MS) {
                    improvement = false;
                    time_limit_exceeded = true;
                    continue;
                }
            }

            if (client_idx >= num_clients)
                continue;

            //DLB!!
            if (dlb_[client_idx])
                continue;


            int u = client_idx + 2; //id
            int g_u = genotype[client_idx]; // group(u)

            if (g_u < 0 || g_u >= num_groups)
                continue;

            bool move_made = false;


            double cost_source_before = route_costs_[g_u];

            double cost_source_after = SimulateRouteCostWithRemoval(g_u, u);
            double source_delta = cost_source_after - cost_source_before;

            candidate_groups_.clear();


            const auto& my_neighbors = geometry_->GetNeighbors(client_idx);
            int checked_count = 0;
            int max_neighbors = Config::NUM_NEIGHBORS;

            //for large neighborhoods, limit the number of neighbors considered
            if (num_clients > 3000)
                max_neighbors = 10;
            else if (num_clients > 2000)
                max_neighbors = 15;
            else if (num_clients > 1500)
                max_neighbors = 25;


            size_t limit = std::min(my_neighbors.size(), (size_t)max_neighbors + 2);
            for (size_t n_i = 0; n_i < limit; ++n_i) {
                if (checked_count > max_neighbors) continue;

                int neighbor_idx = my_neighbors[n_i];
                checked_count++;

                if (neighbor_idx >= num_clients)
                    continue;
                int g_neighbor = genotype[neighbor_idx];
                if (g_neighbor != g_u && g_neighbor >= 0 && g_neighbor < num_groups)
                    candidate_groups_.push_back(g_neighbor);
            }

            //add a bit of randomness
            for (int i = 0; i < 3; ++i)
                candidate_groups_.push_back(rng_() % num_groups);

            std::sort(candidate_groups_.begin(), candidate_groups_.end());
            auto last =
                std::unique(candidate_groups_.begin(), candidate_groups_.end());
            candidate_groups_.erase(last, candidate_groups_.end());


            int best_target_g = -1;
            double best_total_delta = -EPSILON;
            int best_insert_pos = 0;

            for (int target_g : candidate_groups_) {
                if (target_g == g_u)
                    continue;


                if (apply_move_limit) total_moves_checked++;

                double cost_target_before = route_costs_[target_g];
                const auto& route_tgt = vnd_routes_[target_g];

                //find insertion position by rank
                int rank_u = customer_ranks_[u];
                auto it_ins = std::upper_bound(
                    route_tgt.begin(), route_tgt.end(), rank_u,
                    [&](int r, int id) { return r < customer_ranks_[id]; });
                int ins_pos = (int)std::distance(route_tgt.begin(), it_ins);

                double target_delta;

                // O(1) delta calculation
                double fast_target_delta = CalculateFastInsertionDelta(u, target_g, ins_pos);
                if (source_delta + fast_target_delta >= best_total_delta)
                    continue;

                if (IsSafeMove(target_g, u)) {
                    target_delta = fast_target_delta;
                }
                else {
                    double cost_target_after =
                        SimulateRouteCostWithInsert(target_g, u, ins_pos);
                    target_delta = cost_target_after - cost_target_before;
                }

                double total_delta = source_delta + target_delta;

                //better delta wins
                bool accept_move = (total_delta < best_total_delta);



                if (accept_move) {
                    best_total_delta = total_delta;
                    best_target_g = target_g;
                    best_insert_pos = ins_pos;
                }
            }

            if (best_target_g != -1) {
                int old_route = g_u;
                int new_route = best_target_g;

                auto& r_src = vnd_routes_[old_route];
                auto it_rem = std::find(r_src.begin(), r_src.end(), u);
                if (it_rem != r_src.end())
                {
                    r_src.erase(it_rem);
                }
                vnd_loads_[old_route] -= evaluator_->GetDemand(u);

                auto& r_dst = vnd_routes_[new_route];
                auto it_ins = std::upper_bound(
                    r_dst.begin(), r_dst.end(), customer_ranks_[u], [&](int r, int id) { return r < customer_ranks_[id]; });
                r_dst.insert(it_ins, u);
                vnd_loads_[new_route] += evaluator_->GetDemand(u);

                genotype[client_idx] = new_route;

                UpdatePositionsAfterMove(u, old_route, new_route);

                improvement = true;
                any_change = true;
                move_made = true;

                dlb_[client_idx] = false;

                for (int n_idx : geometry_->GetNeighbors(client_idx)) {
                    if (n_idx < (int)dlb_.size())
                        dlb_[n_idx] = false;
                }
            }

            if (!move_made) {

                if (allow_swap) {
                    //swap
                    double best_swap_delta = -EPSILON;
                    int best_swap_target_g = -1;
                    int best_swap_v = -1;

                    const auto& neighbors = geometry_->GetNeighbors(client_idx);

                    const auto& pos_u = positions_[client_idx];
                    int prev_u = (pos_u.prev_client > 0) ? (pos_u.prev_client - 1) : 0;
                    int next_u = (pos_u.next_client > 0) ? (pos_u.next_client - 1) : 0;
                    int u_mat = u - 1;

                    for (int n_idx : neighbors) {
                        if (n_idx >= num_clients)
                            continue;
                        int target_g = genotype[n_idx];
                        if (target_g == g_u || target_g < 0 || target_g >= num_groups)
                            continue;

                        int v = n_idx + 2;

                        const auto& pos_v = positions_[n_idx];
                        int prev_v = (pos_v.prev_client > 0) ? (pos_v.prev_client - 1) : 0;
                        int next_v = (pos_v.next_client > 0) ? (pos_v.next_client - 1) : 0;
                        int v_mat = v - 1;

                        double delta_src, delta_tgt;

                        if (fast_matrix_) {

                            double old_src = fast_matrix_[prev_u * matrix_dim_ + u_mat] + fast_matrix_[u_mat * matrix_dim_ + next_u];
                            double new_src = fast_matrix_[prev_u * matrix_dim_ + v_mat] + fast_matrix_[v_mat * matrix_dim_ + next_u];
                            delta_src = new_src - old_src;

                            // delta for target route
                            double old_tgt = fast_matrix_[prev_v * matrix_dim_ + v_mat] + fast_matrix_[v_mat * matrix_dim_ + next_v];
                            double new_tgt = fast_matrix_[prev_v * matrix_dim_ + u_mat] + fast_matrix_[u_mat * matrix_dim_ + next_v];
                            delta_tgt = new_tgt - old_tgt;
                        }
                        else {
                            double old_src = evaluator_->GetDist(prev_u, u_mat) + evaluator_->GetDist(u_mat, next_u);
                            double new_src = evaluator_->GetDist(prev_u, v_mat) + evaluator_->GetDist(v_mat, next_u);
                            delta_src = new_src - old_src;

                            double old_tgt = evaluator_->GetDist(prev_v, v_mat) + evaluator_->GetDist(v_mat, next_v);
                            double new_tgt = evaluator_->GetDist(prev_v, u_mat) + evaluator_->GetDist(u_mat, next_v);
                            delta_tgt = new_tgt - old_tgt;
                        }

                        double swap_delta = delta_src + delta_tgt;

                        if (swap_delta < best_swap_delta) {
                            best_swap_delta = swap_delta;
                            best_swap_target_g = target_g;
                            best_swap_v = v;
                        }
                    }

                    //execute the best swap found
                    if (best_swap_target_g != -1 && best_swap_v != -1) {
                        int v = best_swap_v;
                        int v_idx = v - 2;
                        int target_g = best_swap_target_g;

                        if (v_idx < 0 || v_idx >= num_clients)
                            continue;

                        auto& r_src = vnd_routes_[g_u];
                        auto it_u_rem = std::find(r_src.begin(), r_src.end(), u);
                        if (it_u_rem == r_src.end())
                            continue;
                        r_src.erase(it_u_rem);

                        auto it_v_ins = std::upper_bound(
                            r_src.begin(), r_src.end(), customer_ranks_[v],
                            [&](int r, int id) { return r < customer_ranks_[id]; });
                        r_src.insert(it_v_ins, v);

                        auto& r_tgt = vnd_routes_[target_g];
                        auto it_v_rem = std::find(r_tgt.begin(), r_tgt.end(), v);
                        if (it_v_rem == r_tgt.end()) {
                            auto it_v_back = std::find(r_src.begin(), r_src.end(), v);
                            if (it_v_back != r_src.end())
                                r_src.erase(it_v_back);
                            auto it_u_back = std::upper_bound(
                                r_src.begin(), r_src.end(), customer_ranks_[u],
                                [&](int r, int id) { return r < customer_ranks_[id]; });
                            r_src.insert(it_u_back, u);
                            continue;
                        }
                        r_tgt.erase(it_v_rem);

                        auto it_u_ins = std::upper_bound(
                            r_tgt.begin(), r_tgt.end(), customer_ranks_[u],
                            [&](int r, int id) { return r < customer_ranks_[id]; });
                        r_tgt.insert(it_u_ins, u);

                        int demand_u = evaluator_->GetDemand(u);
                        int demand_v = evaluator_->GetDemand(v);
                        vnd_loads_[g_u] += demand_v - demand_u;
                        vnd_loads_[target_g] += demand_u - demand_v;


                        genotype[client_idx] = target_g;
                        genotype[v_idx] = g_u;

                        improvement = true;
                        any_change = true;
                        move_made = true;


                        dlb_[client_idx] = false;
                        if (v_idx < (int)dlb_.size())
                            dlb_[v_idx] = false;
                        for (int n_idx : geometry_->GetNeighbors(client_idx)) {
                            if (n_idx < (int)dlb_.size())
                                dlb_[n_idx] = false;
                        }
                        for (int n_idx : geometry_->GetNeighbors(v_idx)) {
                            if (n_idx < (int)dlb_.size())
                                dlb_[n_idx] = false;
                        }


                        route_costs_[g_u] += best_swap_delta / 2;
                        route_costs_[target_g] += best_swap_delta / 2;

                        //update positions
                        const auto& r_tgt_new = vnd_routes_[target_g];
                        for (int pos = 0; pos < (int)r_tgt_new.size(); ++pos) {
                            int cid = r_tgt_new[pos];
                            int cidx = cid - 2;
                            if (cidx >= 0 && cidx < num_clients) {
                                positions_[cidx].route_id = target_g;
                                positions_[cidx].position = pos;
                                positions_[cidx].prev_client =
                                    (pos > 0) ? r_tgt_new[pos - 1] : -1;
                                positions_[cidx].next_client =
                                    (pos < (int)r_tgt_new.size() - 1) ? r_tgt_new[pos + 1] : -1;
                            }
                        }

                        //update source route positions
                        const auto& r_src_new = vnd_routes_[g_u];
                        for (int pos = 0; pos < (int)r_src_new.size(); ++pos) {
                            int cid = r_src_new[pos];
                            int cidx = cid - 2;
                            if (cidx >= 0 && cidx < num_clients) {
                                positions_[cidx].route_id = g_u;
                                positions_[cidx].position = pos;
                                positions_[cidx].prev_client =
                                    (pos > 0) ? r_src_new[pos - 1] : -1;
                                positions_[cidx].next_client =
                                    (pos < (int)r_src_new.size() - 1) ? r_src_new[pos + 1] : -1;
                            }
                        }
                    }
                }

                if (!move_made) {
                    dlb_[client_idx] = true;
                }
            }
        }
    }


    if (allow_3swap && any_change) {
        if (Try3Swap(genotype)) {
            any_change = true;
            BuildPositions();
        }
    }

    if (allow_4swap && any_change) {
        if (Try4Swap(genotype)) {
            any_change = true;
            BuildPositions();
        }
    }

    if (allow_ejection) {
        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        const double EJECTION_PROBABILITY = Config::EJECTION_PROBABILITY;

        for (int client_idx = 0; client_idx < num_clients; ++client_idx) {
            if (prob_dist(rng_) > EJECTION_PROBABILITY)
                continue;

            if (dlb_[client_idx])
                continue;

            int ejection_depth = 2;

            if (allow_4swap && num_clients < 2000) {
                ejection_depth = 3;
            }

            if (TryEjectionChain(genotype, client_idx, ejection_depth)) {
                any_change = true;
                improvement = true;

                dlb_[client_idx] = false;
                for (int n_idx : geometry_->GetNeighbors(client_idx)) {
                    if (n_idx < (int)dlb_.size())
                        dlb_[n_idx] = false;
                }
            }
        }
    }

    bool is_subproblem = (int)client_indices_.size() < num_clients && num_clients > 500;

    std::uniform_real_distribution<double> pr_prob(0.0, 1.0);
    bool should_try_pr = !is_subproblem && ((!any_change) || (pr_prob(rng_) < Config::PATH_RELINK_PROBABILITY));

    if (should_try_pr && !guide_solution_.empty() &&
        guide_solution_.size() == genotype.size()) {

        double current_total = 0.0;
        for (int g = 0; g < num_groups; ++g) {
            current_total += route_costs_[g];
        }


        double guide_cost = 0.0;
        std::vector<std::vector<int>> guide_routes(num_groups);
        for (int i = 0; i < num_clients; ++i) {
            int g = guide_solution_[i];
            if (g >= 0 && g < num_groups) {
                guide_routes[g].push_back(i + 2);
            }
        }
        for (const auto& r : guide_routes) {
            guide_cost += SimulateRouteCost(r);
        }

        if (current_total < guide_cost * 1.15) {

            if (TryPathRelinking(genotype, current_total, guide_solution_)) {
                any_change = true;
            }
        }
    }



    return any_change;
}

bool LocalSearch::RunVND(Individual& ind, bool heavy_mode) {
    // legacy methgod
    int max_iter = heavy_mode ? 50 : 10;
    return RunVND(ind, max_iter, heavy_mode, heavy_mode, heavy_mode, heavy_mode);
}

bool LocalSearch::RunVND(Individual& ind, int max_iter, bool allow_swap, bool allow_3swap, bool allow_ejection, bool allow_4swap, bool unlimited_moves) {
    std::vector<int>& genotype = ind.AccessGenotype();
    if (genotype.empty())
        return false;


    int num_clients = static_cast<int>(genotype.size());
    int num_groups = evaluator_->GetNumGroups();


    client_indices_.resize(num_clients);
    std::iota(client_indices_.begin(), client_indices_.end(), 0);

    //shuffle for better exploration
    std::shuffle(client_indices_.begin(), client_indices_.end(), rng_);

    for (auto& r : vnd_routes_)
        r.clear();
    std::fill(vnd_loads_.begin(), vnd_loads_.end(), 0.0);

    for (int i = 0; i < num_clients; ++i) {
        int u = i + 2;
        int g = genotype[i];
        if (g >= 0 && g < num_groups) {
            vnd_routes_[g].push_back(u);
            vnd_loads_[g] += evaluator_->GetDemand(u);
        }
    }


    for (auto& r : vnd_routes_) {
        std::sort(r.begin(), r.end(), [&](int a, int b) {
            return customer_ranks_[a] < customer_ranks_[b];
            });
    }


    BuildPositions();

    return OptimizeActiveSet(ind, max_iter, allow_swap, allow_3swap,
        allow_ejection, allow_4swap, unlimited_moves);
}




bool LocalSearch::RunHugeInstanceVND(Individual& ind, int tier) {
    std::vector<int>& genotype = ind.AccessGenotype();
    if (genotype.empty()) return false;

    int num_clients = static_cast<int>(genotype.size());
    int num_groups = evaluator_->GetNumGroups();

    //tier system with different limits
    int max_iter;
    int neighbor_limit;
    int move_limit;
    int time_limit_ms;
    bool allow_swap;
    bool allow_3swap;
    bool allow_ejection;

    switch (tier) {
    case 0:
        max_iter = 2;
        neighbor_limit = 6;
        move_limit = 5000;
        time_limit_ms = 30;
        allow_swap = false;
        allow_3swap = false;
        allow_ejection = false;
        break;
    case 1:
        max_iter = 5;
        neighbor_limit = 10;
        move_limit = 100000;
        time_limit_ms = 75;
        allow_swap = true;
        allow_3swap = false;
        allow_ejection = false;
        break;
    case 2:
    default:
        max_iter = 50;
        neighbor_limit = 30;
        move_limit = 5000000;
        time_limit_ms = 2000;
        allow_swap = true;
        allow_3swap = true;
        allow_ejection = true;
        break;
    }

    //prepare routes
    for (auto& r : vnd_routes_) r.clear();
    std::fill(vnd_loads_.begin(), vnd_loads_.end(), 0.0);

    for (int i = 0; i < num_clients; ++i) {
        int u = i + 2;
        int g = genotype[i];
        if (g >= 0 && g < num_groups) {
            vnd_routes_[g].push_back(u);
            vnd_loads_[g] += evaluator_->GetDemand(u);
        }
    }

    // sort routes by rank
    for (auto& r : vnd_routes_) {
        std::sort(r.begin(), r.end(), [&](int a, int b) {
            return customer_ranks_[a] < customer_ranks_[b];
            });
    }


    BuildPositions();
    client_indices_.resize(num_clients);
    std::iota(client_indices_.begin(), client_indices_.end(), 0);
    std::shuffle(client_indices_.begin(), client_indices_.end(), rng_);

    // reset DLB
    if (dlb_.size() != genotype.size()) {
        dlb_.assign(num_clients, false);
    }
    else {
        ResetDLB();
    }


    const double EPSILON = 1e-4;
    bool any_change = false;
    bool improvement = true;
    int iter = 0;
    int total_moves_checked = 0;
    auto vnd_start_time = std::chrono::high_resolution_clock::now();
    int time_check_counter = 0;

    while (improvement && iter < max_iter) {
        improvement = false;
        iter++;

        if (total_moves_checked > move_limit) break;

        for (int client_idx : client_indices_) {

            if ((++time_check_counter % 50) == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - vnd_start_time).count();
                if (elapsed_ms > time_limit_ms) {
                    improvement = false;
                    break;
                }
            }

            if (client_idx >= num_clients) continue;
            if (dlb_[client_idx]) continue;

            int u = client_idx + 2;
            int g_u = genotype[client_idx];
            if (g_u < 0 || g_u >= num_groups) continue;


            candidate_groups_.clear();
            const auto& neighbors = geometry_->GetNeighbors(client_idx);
            int checked = 0;
            for (int n_idx : neighbors) {
                if (checked++ > neighbor_limit) break;
                if (n_idx >= num_clients) continue;
                int g_n = genotype[n_idx];
                if (g_n != g_u && g_n >= 0 && g_n < num_groups) {
                    candidate_groups_.push_back(g_n);
                }
            }

            candidate_groups_.push_back(rng_() % num_groups);
            candidate_groups_.push_back(rng_() % num_groups);

            std::sort(candidate_groups_.begin(), candidate_groups_.end());
            auto last = std::unique(candidate_groups_.begin(), candidate_groups_.end());
            candidate_groups_.erase(last, candidate_groups_.end());

            double cost_src_before = route_costs_[g_u];
            double cost_src_after = SimulateRouteCostWithRemoval(g_u, u);
            double source_delta = cost_src_after - cost_src_before;

            int best_target_g = -1;
            double best_total_delta = -EPSILON;
            int best_insert_pos = 0;

            for (int target_g : candidate_groups_) {
                if (target_g == g_u) continue;
                total_moves_checked++;

                const auto& route_tgt = vnd_routes_[target_g];
                int rank_u = customer_ranks_[u];
                auto it_ins = std::upper_bound(route_tgt.begin(), route_tgt.end(), rank_u,
                    [&](int r, int id) { return r < customer_ranks_[id]; });
                int ins_pos = (int)std::distance(route_tgt.begin(), it_ins);

                double fast_delta = CalculateFastInsertionDelta(u, target_g, ins_pos);
                if (source_delta + fast_delta >= best_total_delta) continue;

                double target_delta;
                if (IsSafeMove(target_g, u)) {
                    target_delta = fast_delta;
                }
                else {
                    double cost_tgt_after = SimulateRouteCostWithInsert(target_g, u, ins_pos);
                    target_delta = cost_tgt_after - route_costs_[target_g];
                }

                double total_delta = source_delta + target_delta;
                if (total_delta < best_total_delta) {
                    best_total_delta = total_delta;
                    best_target_g = target_g;
                    best_insert_pos = ins_pos;
                }
            }

            // execute the best move
            if (best_target_g != -1) {
                auto& r_src = vnd_routes_[g_u];
                auto it_rem = std::find(r_src.begin(), r_src.end(), u);
                if (it_rem != r_src.end()) r_src.erase(it_rem);
                vnd_loads_[g_u] -= evaluator_->GetDemand(u);

                auto& r_dst = vnd_routes_[best_target_g];
                auto it_ins = std::upper_bound(r_dst.begin(), r_dst.end(), customer_ranks_[u],
                    [&](int r, int id) { return r < customer_ranks_[id]; });
                r_dst.insert(it_ins, u);
                vnd_loads_[best_target_g] += evaluator_->GetDemand(u);

                genotype[client_idx] = best_target_g;
                route_costs_[g_u] = SimulateRouteCost(r_src);
                route_costs_[best_target_g] = SimulateRouteCost(r_dst);
                UpdatePositionsAfterMove(u, g_u, best_target_g);

                improvement = true;
                any_change = true;
                dlb_[client_idx] = false;
                for (int n_idx : geometry_->GetNeighbors(client_idx)) {
                    if (n_idx < (int)dlb_.size()) dlb_[n_idx] = false;
                }
            }
            else {
                dlb_[client_idx] = true;
            }
        }
    }


    if (allow_swap) {
        for (int client_idx : client_indices_) {
            if ((++time_check_counter % 50) == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - vnd_start_time).count();
                if (elapsed_ms > time_limit_ms) break;
            }

            if (client_idx >= num_clients) continue;
            if (dlb_[client_idx]) continue;

            int u = client_idx + 2;
            int g_u = genotype[client_idx];
            if (g_u < 0 || g_u >= num_groups) continue;

            double best_swap_delta = -EPSILON;
            int best_swap_target_g = -1;
            int best_swap_v = -1;

            const auto& neighbors = geometry_->GetNeighbors(client_idx);

            const auto& pos_u = positions_[client_idx];
            int prev_u = (pos_u.prev_client > 0) ? (pos_u.prev_client - 1) : 0;
            int next_u = (pos_u.next_client > 0) ? (pos_u.next_client - 1) : 0;
            int u_mat = u - 1;

            int neighbors_checked = 0;
            for (int n_idx : neighbors) {
                if (neighbors_checked++ > neighbor_limit) break;
                if (n_idx >= num_clients) continue;
                int target_g = genotype[n_idx];
                if (target_g == g_u || target_g < 0 || target_g >= num_groups) continue;

                int v = n_idx + 2;
                const auto& pos_v = positions_[n_idx];
                int prev_v = (pos_v.prev_client > 0) ? (pos_v.prev_client - 1) : 0;
                int next_v = (pos_v.next_client > 0) ? (pos_v.next_client - 1) : 0;
                int v_mat = v - 1;

                double delta_src, delta_tgt;

                if (fast_matrix_) {
                    double old_src = fast_matrix_[prev_u * matrix_dim_ + u_mat] + fast_matrix_[u_mat * matrix_dim_ + next_u];
                    double new_src = fast_matrix_[prev_u * matrix_dim_ + v_mat] + fast_matrix_[v_mat * matrix_dim_ + next_u];
                    delta_src = new_src - old_src;

                    double old_tgt = fast_matrix_[prev_v * matrix_dim_ + v_mat] + fast_matrix_[v_mat * matrix_dim_ + next_v];
                    double new_tgt = fast_matrix_[prev_v * matrix_dim_ + u_mat] + fast_matrix_[u_mat * matrix_dim_ + next_v];
                    delta_tgt = new_tgt - old_tgt;
                }
                else {
                    double old_src = evaluator_->GetDist(prev_u, u_mat) + evaluator_->GetDist(u_mat, next_u);
                    double new_src = evaluator_->GetDist(prev_u, v_mat) + evaluator_->GetDist(v_mat, next_u);
                    delta_src = new_src - old_src;

                    double old_tgt = evaluator_->GetDist(prev_v, v_mat) + evaluator_->GetDist(v_mat, next_v);
                    double new_tgt = evaluator_->GetDist(prev_v, u_mat) + evaluator_->GetDist(u_mat, next_v);
                    delta_tgt = new_tgt - old_tgt;
                }

                double swap_delta = delta_src + delta_tgt;

                if (swap_delta < best_swap_delta) {

                    bool safe_swap = true;
                    if (!evaluator_->HasDistanceConstraint()) {
                        int demand_u = evaluator_->GetDemand(u);
                        int demand_v = evaluator_->GetDemand(v);
                        int capacity = evaluator_->GetCapacity();
                        if (demand_v > demand_u) {
                            if (max_cumulative_load_[g_u] + (demand_v - demand_u) > capacity) safe_swap = false;
                        }
                        else if (demand_u > demand_v) {
                            if (max_cumulative_load_[target_g] + (demand_u - demand_v) > capacity) safe_swap = false;
                        }
                    }
                    else {
                        safe_swap = false;
                    }

                    if (safe_swap) {
                        best_swap_delta = swap_delta;
                        best_swap_target_g = target_g;
                        best_swap_v = v;
                    }
                }
            }

            if (best_swap_target_g != -1 && best_swap_v != -1) {
                int v = best_swap_v;
                int v_idx = v - 2;
                int target_g = best_swap_target_g;

                auto& r_src = vnd_routes_[g_u];
                auto it_u_rem = std::find(r_src.begin(), r_src.end(), u);
                if (it_u_rem != r_src.end()) r_src.erase(it_u_rem);

                auto it_v_ins = std::upper_bound(r_src.begin(), r_src.end(), customer_ranks_[v], [&](int r, int id) { return r < customer_ranks_[id]; });
                r_src.insert(it_v_ins, v);

                auto& r_tgt = vnd_routes_[target_g];
                auto it_v_rem = std::find(r_tgt.begin(), r_tgt.end(), v);
                if (it_v_rem != r_tgt.end()) r_tgt.erase(it_v_rem);

                auto it_u_ins = std::upper_bound(r_tgt.begin(), r_tgt.end(), customer_ranks_[u], [&](int r, int id) { return r < customer_ranks_[id]; });
                r_tgt.insert(it_u_ins, u);

                int demand_u = evaluator_->GetDemand(u);
                int demand_v = evaluator_->GetDemand(v);
                vnd_loads_[g_u] += demand_v - demand_u;
                vnd_loads_[target_g] += demand_u - demand_v;

                genotype[client_idx] = target_g;
                genotype[v_idx] = g_u;

                route_costs_[g_u] = SimulateRouteCost(r_src);
                route_costs_[target_g] = SimulateRouteCost(r_tgt);

                {
                    int cap = evaluator_->GetCapacity();
                    int ml = 0, cl = 0;
                    for (int c : r_src) {
                        int d = evaluator_->GetDemand(c);
                        if (cl + d > cap) cl = 0;
                        cl += d;
                        if (cl > ml) ml = cl;
                    }
                    max_cumulative_load_[g_u] = ml;

                    ml = 0; cl = 0;
                    for (int c : r_tgt) {
                        int d = evaluator_->GetDemand(c);
                        if (cl + d > cap) cl = 0;
                        cl += d;
                        if (cl > ml) ml = cl;
                    }
                    max_cumulative_load_[target_g] = ml;
                }

                UpdatePositionsAfterMove(u, g_u, target_g);

                any_change = true;
                dlb_[client_idx] = false;
                if (v_idx < (int)dlb_.size()) dlb_[v_idx] = false;
                for (int n_idx : neighbors) {
                    if (n_idx < (int)dlb_.size()) dlb_[n_idx] = false;
                }
                for (int n_idx : geometry_->GetNeighbors(v_idx)) {
                    if (n_idx < (int)dlb_.size()) dlb_[n_idx] = false;
                }
            }
        }
    }

    if (allow_3swap && any_change) {
        if (Try3Swap(genotype)) {
            any_change = true;
        }
    }


    if (allow_ejection && any_change) {
        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        for (int i = 0; i < num_clients; ++i) {
            if (prob_dist(rng_) > 0.10) continue;
            if (dlb_[i]) continue;
            if (TryEjectionChain(genotype, i, 2)) {
                any_change = true;
            }
        }
    }

    return any_change;
}

bool LocalSearch::RunFullVND(Individual& ind, bool allow_swap) {
    int num_clients = static_cast<int>(ind.GetGenotype().size());


    client_indices_.resize(num_clients);
    std::iota(client_indices_.begin(), client_indices_.end(), 0);


    int max_iter = 20;
    if (num_clients > Config::HUGE_INSTANCE_THRESHOLD) {
        max_iter = 5;
    }
    return OptimizeActiveSet(ind, max_iter, allow_swap, false);
}


void LocalSearch::BuildPositions() {
    int num_clients = evaluator_->GetSolutionSize();
    int num_groups = evaluator_->GetNumGroups();

    positions_.resize(num_clients);
    route_costs_.resize(num_groups);

    // init all positions
    for (int i = 0; i < num_clients; ++i) {
        positions_[i].prev_client = -1;
        positions_[i].next_client = -1;
        positions_[i].route_id = -1;
        positions_[i].position = -1;
    }

    for (int g = 0; g < num_groups; ++g) {
        const auto& route = vnd_routes_[g];
        int route_size = static_cast<int>(route.size());

        for (int pos = 0; pos < route_size; ++pos) {
            int client_id = route[pos];
            int client_idx = client_id - 2;

            if (client_idx >= 0 && client_idx < num_clients) {
                positions_[client_idx].route_id = g;
                positions_[client_idx].position = pos;
                positions_[client_idx].prev_client = (pos > 0) ? route[pos - 1] : -1;
                positions_[client_idx].next_client =
                    (pos < route_size - 1) ? route[pos + 1] : -1;
            }
        }


        route_costs_[g] = SimulateRouteCost(route);
    }

    BuildCumulativeLoads();
}

void LocalSearch::BuildCumulativeLoads() {
    int num_groups = evaluator_->GetNumGroups();
    int capacity = evaluator_->GetCapacity();

    max_cumulative_load_.resize(num_groups);

    for (int g = 0; g < num_groups; ++g) {
        const auto& route = vnd_routes_[g];
        int max_load = 0;
        int current_load = 0;

        for (int client_id : route) {
            int demand = evaluator_->GetDemand(client_id);

            // Check if we would return to depot
            if (current_load + demand > capacity) {
                current_load = 0;
            }

            current_load += demand;
            if (current_load > max_load) {
                max_load = current_load;
            }
        }

        max_cumulative_load_[g] = max_load;
    }
}

bool LocalSearch::IsSafeMove(int target_route, int client_id) const {
    if (target_route < 0 || target_route >= (int)max_cumulative_load_.size())
        return false;

    //again, distance?
    if (evaluator_->HasDistanceConstraint())
        return false;

    int demand = evaluator_->GetDemand(client_id);
    int capacity = evaluator_->GetCapacity();

    return (max_cumulative_load_[target_route] + demand <= capacity);
}


double LocalSearch::CalculateFastInsertionDelta(int client_id, int target_route, int insert_pos) const {
    if (target_route < 0 || target_route >= (int)vnd_routes_.size())
        return 1e30;

    const auto& route = vnd_routes_[target_route];
    int route_size = static_cast<int>(route.size());
    int curr_idx = client_id - 1; // matrix index

    int prev_idx =
        (insert_pos > 0) ? (route[insert_pos - 1] - 1) : 0;
    int next_idx =
        (insert_pos < route_size) ? (route[insert_pos] - 1) : 0;

    // delta = dist(prev, new) + dist(new, next) - dist(prev, next)
    double old_edge, new_edges;

    if (fast_matrix_) {
        old_edge = fast_matrix_[prev_idx * matrix_dim_ + next_idx];
        new_edges = fast_matrix_[prev_idx * matrix_dim_ + curr_idx] +
            fast_matrix_[curr_idx * matrix_dim_ + next_idx];
    }
    else {
        old_edge = evaluator_->GetDist(prev_idx, next_idx);
        new_edges = evaluator_->GetDist(prev_idx, curr_idx) +
            evaluator_->GetDist(curr_idx, next_idx);
    }

    return new_edges - old_edge;
}

double LocalSearch::SimulateRouteCostWithInsert(int target_route, int client_id, int insert_pos) const {
    if (target_route < 0 || target_route >= (int)vnd_routes_.size())
        return 1e30;

    const auto& route = vnd_routes_[target_route];
    int capacity = evaluator_->GetCapacity();
    int route_size = static_cast<int>(route.size());
    bool check_dist = evaluator_->HasDistanceConstraint();
    double max_dist = evaluator_->GetMaxDistance();

    double total_cost = 0.0;
    int current_load = 0;
    double current_segment_dist = 0.0;
    int last_idx = 0; // depo


    if (fast_matrix_) {
        for (int pos = 0; pos <= route_size; ++pos) {

            int current_customer_id = -1;


            if (pos == insert_pos) {
                current_customer_id = client_id;
            }


            int iterations = (pos == insert_pos) ? 2 : 1;

            if (pos == insert_pos) {
                int demand = evaluator_->GetDemand(client_id);
                int customer_idx = client_id - 1;

                if (current_load + demand > capacity) {
                    total_cost += fast_matrix_[last_idx * matrix_dim_ + 0];
                    last_idx = 0;
                    current_load = 0;
                    current_segment_dist = 0.0;
                }

                double d_travel = fast_matrix_[last_idx * matrix_dim_ + customer_idx];

                //distance?
                if (check_dist) {
                    double d_return = fast_matrix_[customer_idx * matrix_dim_ + 0];
                    if (current_segment_dist + d_travel + d_return > max_dist) {
                        if (last_idx != 0) {
                            total_cost += fast_matrix_[last_idx * matrix_dim_ + 0];
                            last_idx = 0;
                            current_load = 0;
                            current_segment_dist = 0.0;
                            d_travel = fast_matrix_[0 * matrix_dim_ + customer_idx];
                        }
                    }
                }

                total_cost += d_travel;
                current_segment_dist += d_travel;
                current_load += demand;
                last_idx = customer_idx;

                if (pos == route_size)
                    continue;
            }

            if (pos < route_size) {
                int orig_client_id = route[pos];
                int demand = evaluator_->GetDemand(orig_client_id);
                int customer_idx = orig_client_id - 1;

                if (current_load + demand > capacity) {
                    total_cost += fast_matrix_[last_idx * matrix_dim_ + 0];
                    last_idx = 0;
                    current_load = 0;
                    current_segment_dist = 0.0;
                }

                double d_travel = fast_matrix_[last_idx * matrix_dim_ + customer_idx];

                if (check_dist) {
                    double d_return = fast_matrix_[customer_idx * matrix_dim_ + 0];
                    if (current_segment_dist + d_travel + d_return > max_dist) {
                        if (last_idx != 0) {
                            total_cost += fast_matrix_[last_idx * matrix_dim_ + 0];
                            last_idx = 0;
                            current_load = 0;
                            current_segment_dist = 0.0;
                            d_travel = fast_matrix_[0 * matrix_dim_ + customer_idx];
                        }
                    }
                }

                total_cost += d_travel;
                current_segment_dist += d_travel;
                current_load += demand;
                last_idx = customer_idx;
            }
        }

        if (last_idx != 0) {
            total_cost += fast_matrix_[last_idx * matrix_dim_ + 0];
        }
    }
    else {

        for (int pos = 0; pos <= route_size; ++pos) {
            if (pos == insert_pos) {
                int demand = evaluator_->GetDemand(client_id);
                int customer_idx = client_id - 1;

                if (current_load + demand > capacity) {
                    total_cost += evaluator_->GetDist(last_idx, 0);
                    last_idx = 0;
                    current_load = 0;
                    current_segment_dist = 0.0;
                }

                double d_travel = evaluator_->GetDist(last_idx, customer_idx);

                if (check_dist) {
                    double d_return = evaluator_->GetDist(customer_idx, 0);
                    if (current_segment_dist + d_travel + d_return > max_dist) {
                        if (last_idx != 0) {
                            total_cost += evaluator_->GetDist(last_idx, 0);
                            last_idx = 0;
                            current_load = 0;
                            current_segment_dist = 0.0;
                            d_travel = evaluator_->GetDist(0, customer_idx);
                        }
                    }
                }

                total_cost += d_travel;
                current_segment_dist += d_travel;
                current_load += demand;
                last_idx = customer_idx;

                if (pos == route_size)
                    continue;
            }

            if (pos < route_size) {
                int orig_client_id = route[pos];
                int demand = evaluator_->GetDemand(orig_client_id);
                int customer_idx = orig_client_id - 1;

                if (current_load + demand > capacity) {
                    total_cost += evaluator_->GetDist(last_idx, 0);
                    last_idx = 0;
                    current_load = 0;
                    current_segment_dist = 0.0;
                }

                double d_travel = evaluator_->GetDist(last_idx, customer_idx);

                if (check_dist) {
                    double d_return = evaluator_->GetDist(customer_idx, 0);
                    if (current_segment_dist + d_travel + d_return > max_dist) {
                        if (last_idx != 0) {
                            total_cost += evaluator_->GetDist(last_idx, 0);
                            last_idx = 0;
                            current_load = 0;
                            current_segment_dist = 0.0;
                            d_travel = evaluator_->GetDist(0, customer_idx);
                        }
                    }
                }

                total_cost += d_travel;
                current_segment_dist += d_travel;
                current_load += demand;
                last_idx = customer_idx;
            }
        }
        if (last_idx != 0) {
            total_cost += evaluator_->GetDist(last_idx, 0);
        }
    }

    return total_cost;
}

double LocalSearch::SimulateRouteCostWithRemoval(int source_route, int client_id) const {
    if (source_route < 0 || source_route >= (int)vnd_routes_.size())
        return 1e30;

    const auto& route = vnd_routes_[source_route];
    int capacity = evaluator_->GetCapacity();
    bool check_dist = evaluator_->HasDistanceConstraint();
    double max_dist = evaluator_->GetMaxDistance();

    double total_cost = 0.0;
    int current_load = 0;
    double current_segment_dist = 0.0;
    int last_idx = 0;

    if (fast_matrix_) {

        for (int orig_client_id : route) {
            if (orig_client_id == client_id)
                continue;
            int demand = evaluator_->GetDemand(orig_client_id);
            int customer_idx = orig_client_id - 1;

            if (current_load + demand > capacity) {
                total_cost += fast_matrix_[last_idx * matrix_dim_ + 0];
                last_idx = 0;
                current_load = 0;
                current_segment_dist = 0.0;
            }

            double d_travel = fast_matrix_[last_idx * matrix_dim_ + customer_idx];

            if (check_dist) {
                double d_return = fast_matrix_[customer_idx * matrix_dim_ + 0];
                if (current_segment_dist + d_travel + d_return > max_dist) {
                    if (last_idx != 0) {
                        total_cost += fast_matrix_[last_idx * matrix_dim_ + 0];
                        last_idx = 0;
                        current_load = 0;
                        current_segment_dist = 0.0;
                        d_travel = fast_matrix_[0 * matrix_dim_ + customer_idx];
                    }
                }
            }

            total_cost += d_travel;
            current_segment_dist += d_travel;
            current_load += demand;
            last_idx = customer_idx;
        }
        if (last_idx != 0) {
            total_cost += fast_matrix_[last_idx * matrix_dim_ + 0];
        }
    }
    else {

        for (int orig_client_id : route) {
            if (orig_client_id == client_id)
                continue;
            int demand = evaluator_->GetDemand(orig_client_id);
            int customer_idx = orig_client_id - 1;

            if (current_load + demand > capacity) {
                total_cost += evaluator_->GetDist(last_idx, 0);
                last_idx = 0;
                current_load = 0;
                current_segment_dist = 0.0;
            }

            double d_travel = evaluator_->GetDist(last_idx, customer_idx);

            if (check_dist) {
                double d_return = evaluator_->GetDist(customer_idx, 0);
                if (current_segment_dist + d_travel + d_return > max_dist) {
                    if (last_idx != 0) {
                        total_cost += evaluator_->GetDist(last_idx, 0);
                        last_idx = 0;
                        current_load = 0;
                        current_segment_dist = 0.0;
                        d_travel = evaluator_->GetDist(0, customer_idx);
                    }
                }
            }

            total_cost += d_travel;
            current_segment_dist += d_travel;
            current_load += demand;
            last_idx = customer_idx;
        }
        if (last_idx != 0) {
            total_cost += evaluator_->GetDist(last_idx, 0);
        }
    }

    return total_cost;
}

void LocalSearch::UpdatePositionsAfterMove(int client_id, int old_route,
    int new_route) {

    int num_groups = evaluator_->GetNumGroups();
    int num_clients = evaluator_->GetSolutionSize();


    if (old_route >= 0 && old_route < num_groups) {
        const auto& route = vnd_routes_[old_route];
        int route_size = static_cast<int>(route.size());
        for (int pos = 0; pos < route_size; ++pos) {
            int cid = route[pos];
            int cidx = cid - 2;
            if (cidx >= 0 && cidx < num_clients) {
                positions_[cidx].route_id = old_route;
                positions_[cidx].position = pos;
                positions_[cidx].prev_client = (pos > 0) ? route[pos - 1] : -1;
                positions_[cidx].next_client =
                    (pos < route_size - 1) ? route[pos + 1] : -1;
            }
        }
        route_costs_[old_route] = SimulateRouteCost(route);

        int max_load = 0;
        int current_load = 0;
        int capacity = evaluator_->GetCapacity();
        for (int cid : route) {
            int demand = evaluator_->GetDemand(cid);
            if (current_load + demand > capacity)
                current_load = 0;
            current_load += demand;
            if (current_load > max_load)
                max_load = current_load;
        }
        if (old_route < (int)max_cumulative_load_.size())
            max_cumulative_load_[old_route] = max_load;
    }


    if (new_route >= 0 && new_route < num_groups) {
        const auto& route = vnd_routes_[new_route];
        int route_size = static_cast<int>(route.size());
        for (int pos = 0; pos < route_size; ++pos) {
            int cid = route[pos];
            int cidx = cid - 2;
            if (cidx >= 0 && cidx < num_clients) {
                positions_[cidx].route_id = new_route;
                positions_[cidx].position = pos;
                positions_[cidx].prev_client = (pos > 0) ? route[pos - 1] : -1;
                positions_[cidx].next_client =
                    (pos < route_size - 1) ? route[pos + 1] : -1;
            }
        }
        route_costs_[new_route] = SimulateRouteCost(route);


        int max_load = 0;
        int current_load = 0;
        int capacity = evaluator_->GetCapacity();
        for (int cid : route) {
            int demand = evaluator_->GetDemand(cid);
            if (current_load + demand > capacity)
                current_load = 0;
            current_load += demand;
            if (current_load > max_load)
                max_load = current_load;
        }
        if (new_route < (int)max_cumulative_load_.size())
            max_cumulative_load_[new_route] = max_load;
    }
}

double LocalSearch::CalculateRemovalDelta(int client_id) const {
    int client_idx = client_id - 2;
    if (client_idx < 0 || client_idx >= (int)positions_.size())
        return 0.0;

    const auto& pos = positions_[client_idx];
    if (pos.route_id < 0)
        return 0.0;


    int prev_idx = (pos.prev_client > 0) ? (pos.prev_client - 1) : 0;
    int curr_idx = client_id - 1;
    int next_idx = (pos.next_client > 0) ? (pos.next_client - 1) : 0;

    // delta = -dist(prev, curr) - dist(curr, next) + dist(prev, next)
    double old_cost, new_cost;

    if (fast_matrix_) {
        old_cost = fast_matrix_[prev_idx * matrix_dim_ + curr_idx] +
            fast_matrix_[curr_idx * matrix_dim_ + next_idx];
        new_cost = fast_matrix_[prev_idx * matrix_dim_ + next_idx];
    }
    else {
        old_cost = evaluator_->GetDist(prev_idx, curr_idx) +
            evaluator_->GetDist(curr_idx, next_idx);
        new_cost = evaluator_->GetDist(prev_idx, next_idx);
    }

    return new_cost - old_cost; // negative = improved!
}

double LocalSearch::CalculateInsertionDelta(int client_id, int target_route, int& best_insert_pos) const {
    if (target_route < 0 || target_route >= (int)vnd_routes_.size())
        return 1e30;

    const auto& route = vnd_routes_[target_route];

    int rank_u = customer_ranks_[client_id];
    int route_size = static_cast<int>(route.size());

    int pos = 0;
    while (pos < route_size) {
        if (customer_ranks_[route[pos]] > rank_u) break;
        pos++;
    }
    best_insert_pos = pos;


    int prev_idx = (pos > 0) ? (route[pos - 1] - 1) : 0;
    int next_idx = (pos < route_size) ? (route[pos] - 1) : 0;
    int curr_idx = client_id - 1;

    double old_cost, new_cost;

    if (fast_matrix_) {
        old_cost = fast_matrix_[prev_idx * matrix_dim_ + next_idx];
        new_cost = fast_matrix_[prev_idx * matrix_dim_ + curr_idx] +
            fast_matrix_[curr_idx * matrix_dim_ + next_idx];
    }
    else {
        old_cost = evaluator_->GetDist(prev_idx, next_idx);
        new_cost = evaluator_->GetDist(prev_idx, curr_idx) +
            evaluator_->GetDist(curr_idx, next_idx);
    }

    return new_cost - old_cost;
}

bool LocalSearch::WouldOverflow(int target_route, int client_id) const {
    if (target_route < 0 || target_route >= (int)vnd_loads_.size())
        return true;

    int demand = evaluator_->GetDemand(client_id);
    int capacity = evaluator_->GetCapacity();

    return (vnd_loads_[target_route] + demand > capacity);
}


bool LocalSearch::Try3Swap(std::vector<int>& genotype) {
    int num_clients = static_cast<int>(genotype.size());
    if (num_clients < 3)
        return false;

    int num_groups = evaluator_->GetNumGroups();
    const double EPSILON = 1e-4;


    int capacity = evaluator_->GetCapacity();
    double tight_threshold = capacity * 0.90;

    std::vector<int> route_loads(num_groups, 0);
    for (int i = 0; i < num_clients; ++i) {
        int g = genotype[i];
        if (g >= 0 && g < num_groups) {
            route_loads[g] += evaluator_->GetDemand(i + 2);
        }
    }

    std::vector<int> tight_clients;
    for (int i = 0; i < num_clients; ++i) {
        int g = genotype[i];
        if (g >= 0 && g < num_groups && route_loads[g] > tight_threshold) {
            tight_clients.push_back(i);
        }
    }


    int idx1;
    std::uniform_real_distribution<double> d(0.0, 1.0);
    if (!tight_clients.empty() && d(rng_) < 0.70) {
        idx1 = tight_clients[rng_() % tight_clients.size()];
    }
    else {
        idx1 = rng_() % num_clients;
    }


    const auto& neighbors1 = geometry_->GetNeighbors(idx1);
    int idx2 = -1;

    // select idx2
    int g1 = genotype[idx1];
    for (int n_idx : neighbors1) {
        if (n_idx >= 0 && n_idx < num_clients && n_idx != idx1) {
            int g_n = genotype[n_idx];
            if (g_n != g1 && g_n >= 0 && g_n < num_groups) {
                idx2 = n_idx;
                break;
            }
        }
    }
    //fallback any valid neighbor
    if (idx2 == -1 && !neighbors1.empty()) {
        for (int n_idx : neighbors1) {
            if (n_idx >= 0 && n_idx < num_clients && n_idx != idx1) {
                idx2 = n_idx;
                break;
            }
        }
    }
    if (idx2 == -1) return false;

    //select idx3
    int idx3 = -1;
    int g2 = genotype[idx2];


    const auto& neighbors2 = geometry_->GetNeighbors(idx2);
    for (int n_idx : neighbors2) {
        if (n_idx >= 0 && n_idx < num_clients && n_idx != idx1 && n_idx != idx2) {
            int g_n = genotype[n_idx];

            if (g_n != g1 && g_n != g2 && g_n >= 0 && g_n < num_groups) {
                idx3 = n_idx;
                break;
            }
        }
    }

    if (idx3 == -1) {
        for (int n_idx : neighbors2) {
            if (n_idx >= 0 && n_idx < num_clients && n_idx != idx1 && n_idx != idx2) {
                idx3 = n_idx;
                break;
            }
        }
    }

    if (idx3 == -1) {
        for (int n_idx : neighbors1) {
            if (n_idx >= 0 && n_idx < num_clients && n_idx != idx1 && n_idx != idx2) {
                idx3 = n_idx;
                break;
            }
        }
    }
    if (idx3 == -1) return false;

    int g3 = genotype[idx3];


    if (g1 == g2 && g2 == g3)
        return false;


    std::set<int> affected_set = { g1, g2, g3 };
    std::vector<int> affected_groups(affected_set.begin(), affected_set.end());

    std::map<int, std::vector<int>> local_routes;
    const auto& permutation = evaluator_->GetPermutation();

    for (int g : affected_groups) {
        local_routes[g].clear();
    }

    for (int perm_id : permutation) {
        int client_idx = perm_id - 2;
        if (client_idx >= 0 && client_idx < num_clients) {
            int g = genotype[client_idx];
            if (affected_set.count(g)) {
                local_routes[g].push_back(perm_id);
            }
        }
    }

    double current_total = 0.0;
    for (int g : affected_groups) {
        current_total += SimulateRouteCost(local_routes[g]);
    }

    //al 3! perms
    int perms[6][3] = { {g1, g2, g3},
                       {g1, g3, g2}, {g2, g1, g3}, {g2, g3, g1},
                       {g3, g1, g2}, {g3, g2, g1} };

    int best_perm = -1;
    double best_delta = 0.0;

    for (int p = 1; p < 6; ++p) {
        int new_g1 = perms[p][0];
        int new_g2 = perms[p][1];
        int new_g3 = perms[p][2];

        std::map<int, std::vector<int>> temp_routes;
        for (int g : affected_groups) {
            temp_routes[g].clear();
        }

        for (int perm_id : permutation) {
            int client_idx = perm_id - 2;
            if (client_idx >= 0 && client_idx < num_clients) {
                int g = genotype[client_idx];

                if (client_idx == idx1)
                    g = new_g1;
                else if (client_idx == idx2)
                    g = new_g2;
                else if (client_idx == idx3)
                    g = new_g3;

                if (affected_set.count(g)) {
                    temp_routes[g].push_back(perm_id);
                }
            }
        }

        double new_total = 0.0;
        for (int g : affected_groups) {
            new_total += SimulateRouteCost(temp_routes[g]);
        }

        double delta = new_total - current_total;
        if (delta < best_delta - EPSILON) {
            best_delta = delta;
            best_perm = p;
        }
    }

    if (best_perm > 0) {
        genotype[idx1] = perms[best_perm][0];
        genotype[idx2] = perms[best_perm][1];
        genotype[idx3] = perms[best_perm][2];
        return true;
    }

    return false;
}


bool LocalSearch::Try4Swap(std::vector<int>& genotype) {
    // just like 3 but with 4 clients
    int num_clients = static_cast<int>(genotype.size());
    if (num_clients < 4)
        return false;

    int num_groups = evaluator_->GetNumGroups();
    const double EPSILON = 1e-4;


    std::vector<int> idx(4);
    idx[0] = rng_() % num_clients;

    const auto& neighbors0 = geometry_->GetNeighbors(idx[0]);


    int g0 = genotype[idx[0]];
    idx[1] = -1;
    for (int n_idx : neighbors0) {
        if (n_idx >= 0 && n_idx < num_clients && n_idx != idx[0]) {
            int g_n = genotype[n_idx];
            if (g_n != g0 && g_n >= 0 && g_n < num_groups) {
                idx[1] = n_idx;
                break;
            }
        }
    }
    if (idx[1] == -1 && !neighbors0.empty()) {
        for (int n_idx : neighbors0) {
            if (n_idx >= 0 && n_idx < num_clients && n_idx != idx[0]) {
                idx[1] = n_idx;
                break;
            }
        }
    }
    if (idx[1] == -1) return false;


    const auto& neighbors1 = geometry_->GetNeighbors(idx[1]);
    int g1 = genotype[idx[1]];
    idx[2] = -1;
    for (int n_idx : neighbors1) {
        if (n_idx >= 0 && n_idx < num_clients && n_idx != idx[0] && n_idx != idx[1]) {
            int g_n = genotype[n_idx];
            if (g_n != g0 && g_n != g1 && g_n >= 0 && g_n < num_groups) {
                idx[2] = n_idx;
                break;
            }
        }
    }
    if (idx[2] == -1) {
        for (int n_idx : neighbors1) {
            if (n_idx >= 0 && n_idx < num_clients && n_idx != idx[0] && n_idx != idx[1]) {
                idx[2] = n_idx;
                break;
            }
        }
    }
    if (idx[2] == -1) return false;


    const auto& neighbors2 = geometry_->GetNeighbors(idx[2]);
    int g2 = genotype[idx[2]];
    idx[3] = -1;
    for (int n_idx : neighbors2) {
        if (n_idx >= 0 && n_idx < num_clients &&
            n_idx != idx[0] && n_idx != idx[1] && n_idx != idx[2]) {
            int g_n = genotype[n_idx];
            if (g_n != g0 && g_n != g1 && g_n != g2 && g_n >= 0 && g_n < num_groups) {
                idx[3] = n_idx;
                break;
            }
        }
    }
    if (idx[3] == -1) {
        for (int n_idx : neighbors2) {
            if (n_idx >= 0 && n_idx < num_clients &&
                n_idx != idx[0] && n_idx != idx[1] && n_idx != idx[2]) {
                idx[3] = n_idx;
                break;
            }
        }
    }

    if (idx[3] == -1) {
        for (int n_idx : neighbors0) {
            if (n_idx >= 0 && n_idx < num_clients &&
                n_idx != idx[0] && n_idx != idx[1] && n_idx != idx[2]) {
                idx[3] = n_idx;
                break;
            }
        }
    }
    if (idx[3] == -1) return false;

    std::vector<int> g(4);
    for (int i = 0; i < 4; ++i)
        g[i] = genotype[idx[i]];


    bool all_same = (g[0] == g[1] && g[1] == g[2] && g[2] == g[3]);
    if (all_same)
        return false;


    std::set<int> affected_set(g.begin(), g.end());
    std::vector<int> affected_groups(affected_set.begin(), affected_set.end());


    std::map<int, std::vector<int>> local_routes;
    const auto& permutation = evaluator_->GetPermutation();

    for (int grp : affected_groups) {
        local_routes[grp].clear();
    }

    for (int perm_id : permutation) {
        int client_idx = perm_id - 2;
        if (client_idx >= 0 && client_idx < num_clients) {
            int grp = genotype[client_idx];
            if (affected_set.count(grp)) {
                local_routes[grp].push_back(perm_id);
            }
        }
    }

    double current_total = 0.0;
    for (int grp : affected_groups) {
        current_total += SimulateRouteCost(local_routes[grp]);
    }


    int perm_order[4] = { 0, 1, 2, 3 };
    int best_perm_order[4] = { 0, 1, 2, 3 };
    double best_delta = 0.0;
    bool found = false;


    do {

        if (perm_order[0] == 0 && perm_order[1] == 1 && perm_order[2] == 2 && perm_order[3] == 3) {
            continue;
        }

        std::map<int, std::vector<int>> temp_routes;
        for (int grp : affected_groups) {
            temp_routes[grp].clear();
        }

        for (int perm_id : permutation) {
            int client_idx = perm_id - 2;
            if (client_idx >= 0 && client_idx < num_clients) {
                int grp = genotype[client_idx];

                for (int i = 0; i < 4; ++i) {
                    if (client_idx == idx[i]) {
                        grp = g[perm_order[i]];
                        break;
                    }
                }

                if (affected_set.count(grp)) {
                    temp_routes[grp].push_back(perm_id);
                }
            }
        }

        double new_total = 0.0;
        for (int grp : affected_groups) {
            new_total += SimulateRouteCost(temp_routes[grp]);
        }

        double delta = new_total - current_total;
        if (delta < best_delta - EPSILON) {
            best_delta = delta;
            std::copy(perm_order, perm_order + 4, best_perm_order);
            found = true;
        }
    } while (std::next_permutation(perm_order, perm_order + 4));

    if (found) {
        for (int i = 0; i < 4; ++i) {
            genotype[idx[i]] = g[best_perm_order[i]];
        }
        return true;
    }

    return false;
}


bool LocalSearch::TryEjectionChain(std::vector<int>& genotype, int start_client_idx, int max_depth) {
    //multi hop realocation chain
    int num_clients = static_cast<int>(genotype.size());
    int num_groups = evaluator_->GetNumGroups();
    const double EPSILON = 1e-4;

    if (start_client_idx < 0 || start_client_idx >= num_clients)
        return false;

    int start_client_id = start_client_idx + 2;
    int start_group = genotype[start_client_idx];

    if (start_group < 0 || start_group >= num_groups)
        return false;

    //current total cost of potentially affected routes
    double original_total = 0.0;
    for (int g = 0; g < num_groups; ++g) {
        original_total += route_costs_[g];
    }


    struct ChainMove {
        int client_id;
        int from_group;
        int to_group;
        int ejected_client;
    };

    std::vector<ChainMove> best_chain;
    double best_delta = -EPSILON;


    const auto& neighbors1 = geometry_->GetNeighbors(start_client_idx);

    // sample up to 5 target groups
    std::vector<int> target_groups1;
    for (int n_idx : neighbors1) {
        if (n_idx >= num_clients)
            continue;
        int g = genotype[n_idx];
        if (g != start_group && g >= 0 && g < num_groups) {
            target_groups1.push_back(g);
        }
        if (target_groups1.size() >= 5)
            break;
    }
    // add 2 more for randomness
    for (int i = 0; i < 2; ++i) {
        int rg = rng_() % num_groups;
        if (rg != start_group)
            target_groups1.push_back(rg);
    }

    for (int target_g1 : target_groups1) {
        if (target_g1 == start_group)
            continue;

        //find client to eject
        const auto& route1 = vnd_routes_[target_g1];
        if (route1.empty())
            continue;


        double cost_src_after =
            SimulateRouteCostWithRemoval(start_group, start_client_id);


        int rank_u = customer_ranks_[start_client_id];
        auto it_ins = std::upper_bound(
            route1.begin(), route1.end(), rank_u,
            [&](int r, int id) { return r < customer_ranks_[id]; });
        int ins_pos = static_cast<int>(std::distance(route1.begin(), it_ins));

        double cost_tgt_after =
            SimulateRouteCostWithInsert(target_g1, start_client_id, ins_pos);

        double delta1 = (cost_src_after + cost_tgt_after) -
            (route_costs_[start_group] + route_costs_[target_g1]);


        if (delta1 > route_costs_[start_group] * 0.1)
            continue;

        if (max_depth >= 2 && !route1.empty()) {

            std::vector<int> eject_candidates;
            if (route1.size() <= 3) {
                eject_candidates = route1;
            }
            else {
                for (int i = 0; i < 3; ++i) {
                    eject_candidates.push_back(route1[rng_() % route1.size()]);
                }
            }

            for (int eject_id1 : eject_candidates) {
                if (eject_id1 == start_client_id)
                    continue;
                int eject_idx1 = eject_id1 - 2;
                if (eject_idx1 < 0 || eject_idx1 >= num_clients)
                    continue;

                const auto& neighbors2 = geometry_->GetNeighbors(eject_idx1);
                std::vector<int> target_groups2;
                for (int n_idx : neighbors2) {
                    if (n_idx >= num_clients)
                        continue;
                    int g = genotype[n_idx];
                    if (g != target_g1 && g != start_group && g >= 0 && g < num_groups) {
                        target_groups2.push_back(g);
                    }
                    if (target_groups2.size() >= 3)
                        break;
                }

                target_groups2.push_back(start_group);

                for (int target_g2 : target_groups2) {

                    std::vector<int> temp_src = vnd_routes_[start_group];
                    auto it_rem =
                        std::find(temp_src.begin(), temp_src.end(), start_client_id);
                    if (it_rem != temp_src.end())
                        temp_src.erase(it_rem);

                    std::vector<int> temp_tgt1 = vnd_routes_[target_g1];
                    it_rem = std::find(temp_tgt1.begin(), temp_tgt1.end(), eject_id1);
                    if (it_rem != temp_tgt1.end())
                        temp_tgt1.erase(it_rem);

                    auto it_ins1 = std::upper_bound(temp_tgt1.begin(), temp_tgt1.end(),
                        customer_ranks_[start_client_id],
                        [&](int r, int id) { return r < customer_ranks_[id]; });
                    temp_tgt1.insert(it_ins1, start_client_id);

                    std::vector<int> temp_tgt2 = vnd_routes_[target_g2];
                    if (target_g2 == start_group) {
                        temp_tgt2 = temp_src;
                    }
                    auto it_ins2 = std::upper_bound(
                        temp_tgt2.begin(), temp_tgt2.end(), customer_ranks_[eject_id1], [&](int r, int id) { return r < customer_ranks_[id]; });
                    temp_tgt2.insert(it_ins2, eject_id1);

                    // new costs
                    double new_cost_src = SimulateRouteCost(temp_src);
                    double new_cost_tgt1 = SimulateRouteCost(temp_tgt1);
                    double new_cost_tgt2 = (target_g2 == start_group)
                        ? SimulateRouteCost(temp_tgt2)
                        : SimulateRouteCost(temp_tgt2);

                    // delta
                    double old_cost = route_costs_[start_group] + route_costs_[target_g1];
                    if (target_g2 != start_group && target_g2 != target_g1) {
                        old_cost += route_costs_[target_g2];
                    }

                    double new_cost = new_cost_src + new_cost_tgt1;
                    if (target_g2 != start_group) {
                        new_cost += new_cost_tgt2;
                    }

                    double total_delta = new_cost - old_cost;

                    if (total_delta < best_delta) {
                        best_delta = total_delta;
                        best_chain.clear();
                        best_chain.push_back(
                            { start_client_id, start_group, target_g1, eject_id1 });
                        best_chain.push_back({ eject_id1, target_g1, target_g2, -1 });
                    }
                }
            }
        }


        if (delta1 < best_delta) {
            best_delta = delta1;
            best_chain.clear();
            best_chain.push_back({ start_client_id, start_group, target_g1, -1 });
        }
    }


    if (!best_chain.empty() && best_delta < -EPSILON) {
        for (const auto& move : best_chain) {
            int client_idx = move.client_id - 2;
            if (client_idx >= 0 && client_idx < num_clients) {
                genotype[client_idx] = move.to_group;


                auto& r_from = vnd_routes_[move.from_group];
                auto it = std::find(r_from.begin(), r_from.end(), move.client_id);
                if (it != r_from.end())
                    r_from.erase(it);

                auto& r_to = vnd_routes_[move.to_group];
                auto it_ins = std::upper_bound(
                    r_to.begin(), r_to.end(), customer_ranks_[move.client_id],
                    [&](int r, int id) { return r < customer_ranks_[id]; });
                r_to.insert(it_ins, move.client_id);
            }
        }


        std::set<int> affected;
        for (const auto& m : best_chain) {
            affected.insert(m.from_group);
            affected.insert(m.to_group);
        }
        for (int g : affected) {
            route_costs_[g] = SimulateRouteCost(vnd_routes_[g]);
        }

        return true;
    }

    return false;
}


bool LocalSearch::TryPathRelinking(std::vector<int>& genotype, double& current_cost, const std::vector<int>& guide_solution) {
    //the main goal is to connect genotype to guide_solution via a path of intermediate solutions, hoping to find something better
    if (guide_solution.empty() || guide_solution.size() != genotype.size()) {
        return false;
    }

    int num_clients = static_cast<int>(genotype.size());
    int num_groups = evaluator_->GetNumGroups();
    const double EPSILON = 1e-4;

    // differemce indices between genotype and guide_solution
    std::vector<int> diff_indices;
    diff_indices.reserve(num_clients);
    for (int i = 0; i < num_clients; ++i) {
        if (genotype[i] != guide_solution[i]) {
            diff_indices.push_back(i);
        }
    }

    // if solutions are identical, nothing to explore
    if (diff_indices.empty()) {
        return false;
    }

    // best solution found during path exploration
    std::vector<int> best_genotype = genotype;
    double best_cost = current_cost;
    bool found_improvement = false;

    // copy for path exploration
    std::vector<int> working = genotype;

    //initial routes for simulation
    std::vector<std::vector<int>> routes(num_groups);
    for (int i = 0; i < num_clients; ++i) {
        int g = working[i];
        if (g >= 0 && g < num_groups) {
            routes[g].push_back(i + 2);
        }
    }
    for (auto& r : routes) {
        std::sort(r.begin(), r.end(), [&](int a, int b) {
            return customer_ranks_[a] < customer_ranks_[b];
            });
    }

    std::vector<double> costs(num_groups);
    double working_cost = 0.0;
    for (int g = 0; g < num_groups; ++g) {
        costs[g] = SimulateRouteCost(routes[g]);
        working_cost += costs[g];
    }

    // at each step, try to move one client from its current group to the guide's group
    int max_steps = std::min((int)diff_indices.size(), num_clients / 2);

    for (int step = 0; step < max_steps && !diff_indices.empty(); ++step) {
        int best_move_idx = -1;
        double best_move_cost = 1e30;
        double best_move_delta = 1e30;


        for (int di = 0; di < (int)diff_indices.size(); ++di) {
            int client_idx = diff_indices[di];
            int client_id = client_idx + 2;

            int old_group = working[client_idx];
            int new_group = guide_solution[client_idx];

            if (old_group == new_group)
                continue;
            if (old_group < 0 || old_group >= num_groups)
                continue;
            if (new_group < 0 || new_group >= num_groups)
                continue;


            double old_route_cost = costs[old_group];
            std::vector<int> temp_old = routes[old_group];
            auto it = std::find(temp_old.begin(), temp_old.end(), client_id);
            if (it != temp_old.end())
                temp_old.erase(it);
            double new_old_cost = SimulateRouteCost(temp_old);

            //cost of inserting into new route
            double old_new_cost = costs[new_group];
            std::vector<int> temp_new = routes[new_group];
            auto ins_pos = std::upper_bound(
                temp_new.begin(), temp_new.end(), customer_ranks_[client_id],
                [&](int r, int id) { return r < customer_ranks_[id]; });
            temp_new.insert(ins_pos, client_id);
            double new_new_cost = SimulateRouteCost(temp_new);

            double delta =
                (new_old_cost - old_route_cost) + (new_new_cost - old_new_cost);
            double candidate_cost = working_cost + delta;

            //track best move
            if (candidate_cost < best_move_cost) {
                best_move_cost = candidate_cost;
                best_move_delta = delta;
                best_move_idx = di;
            }
        }

        // apply best move
        if (best_move_idx < 0)
            break;

        int client_idx = diff_indices[best_move_idx];
        int client_id = client_idx + 2;
        int old_group = working[client_idx];
        int new_group = guide_solution[client_idx];

        // update working solution
        working[client_idx] = new_group;
        working_cost += best_move_delta;


        auto& r_old = routes[old_group];
        auto it = std::find(r_old.begin(), r_old.end(), client_id);
        if (it != r_old.end())
            r_old.erase(it);

        auto& r_new = routes[new_group];
        auto ins_pos = std::upper_bound(
            r_new.begin(), r_new.end(), customer_ranks_[client_id],
            [&](int r, int id) { return r < customer_ranks_[id]; });
        r_new.insert(ins_pos, client_id);


        costs[old_group] = SimulateRouteCost(r_old);
        costs[new_group] = SimulateRouteCost(r_new);

        diff_indices.erase(diff_indices.begin() + best_move_idx);


        if (working_cost < best_cost - EPSILON) {
            best_cost = working_cost;
            best_genotype = working;
            found_improvement = true;
        }


        if (found_improvement && step % 5 == 0) {

            for (int ci = 0; ci < num_clients; ++ci) {
                int u = ci + 2;
                int g_u = working[ci];
                if (g_u < 0 || g_u >= num_groups)
                    continue;


                double remove_delta = SimulateRouteCost(routes[g_u]) - costs[g_u];

            }
        }
    }


    if (found_improvement) {
        if (current_cost - best_cost > 1.0) {

            prsucc++;
        }
        genotype = best_genotype;
        current_cost = best_cost;

        //rebuild internal structures for consistency
        for (auto& r : vnd_routes_)
            r.clear();
        for (int i = 0; i < num_clients; ++i) {
            int g = genotype[i];
            if (g >= 0 && g < num_groups) {
                vnd_routes_[g].push_back(i + 2);
            }
        }
        for (auto& r : vnd_routes_) {
            std::sort(r.begin(), r.end(), [&](int a, int b) {
                return customer_ranks_[a] < customer_ranks_[b];
                });
        }
        BuildPositions();
    }

    return found_improvement;
}


