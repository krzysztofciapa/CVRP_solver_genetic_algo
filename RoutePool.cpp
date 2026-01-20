#include "RoutePool.hpp"
#include "Constants.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

using namespace std;

namespace LcVRPContest {

    // calculate cost for a single route with capacity/distance constraints
    double RoutePool::CalculateRouteCost(const std::vector<int>& route, const ThreadSafeEvaluator& evaluator) const {
        if (route.empty()) return 0.0;

        const int depot_index = 0;
        const int capacity = evaluator.GetCapacity();
        const double max_distance = evaluator.GetProblemData().HasDistanceConstraint()
            ? evaluator.GetProblemData().GetDistance()
            : -1.0;

        double total_cost = 0.0;
        int current_load = 0;
        double current_distance = 0.0;
        int last_pos = depot_index;

        for (int customer_id : route) {
            int matrix_idx = customer_id - 1;
            int demand = static_cast<int>(evaluator.GetDemand(customer_id));

            // capacity constraint check
            if (current_load + demand > capacity) {
                total_cost += current_distance + evaluator.GetDist(last_pos, depot_index);
                current_load = 0;
                current_distance = 0.0;
                last_pos = depot_index;
            }

            double dist_to_customer = evaluator.GetDist(last_pos, matrix_idx);

            // distance constraint check
            if (max_distance > 0.0) {
                double dist_back = evaluator.GetDist(matrix_idx, depot_index);
                if (current_distance + dist_to_customer + dist_back > max_distance) {
                    if (last_pos != depot_index) {
                        total_cost += current_distance + evaluator.GetDist(last_pos, depot_index);
                        current_load = 0;
                        current_distance = 0.0;
                        last_pos = depot_index;
                    }
                    dist_to_customer = evaluator.GetDist(depot_index, matrix_idx);
                }
            }

            current_distance += dist_to_customer;
            current_load += demand;
            last_pos = matrix_idx;
        }

        //final return to depot
        total_cost += current_distance + evaluator.GetDist(last_pos, depot_index);
        return total_cost;
    }


    uint64_t RoutePool::HashRoute(const std::vector<int>& sorted_route) const {
        uint64_t hash = 14695981039346656037ULL;
        for (int id : sorted_route) {
            hash ^= static_cast<uint64_t>(id);
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    void RoutePool::AddRoutesFromSolution(const std::vector<int>& solution, const ThreadSafeEvaluator& evaluator) {
        const auto& permutation = evaluator.GetPermutation();
        int num_groups = evaluator.GetNumGroups();
        int num_clients = evaluator.GetSolutionSize();
        const int capacity = evaluator.GetCapacity();
        size_t mask_words = (num_clients + 63) / 64;

        std::vector<std::vector<int>> group_routes(num_groups);
        for (int customer_id : permutation) {
            int gene_idx = customer_id - 2;
            if (gene_idx < 0 || gene_idx >= (int)solution.size()) continue;

            int group = solution[gene_idx];
            if (group >= 0 && group < num_groups) {
                group_routes[group].push_back(customer_id);
            }
        }

        std::lock_guard<std::mutex> lock(mutex_);

        for (const auto& route : group_routes) {
            if (route.size() < Config::MIN_ROUTE_SIZE_FOR_POOL) continue;

            std::vector<int> sorted_route = route;
            std::sort(sorted_route.begin(), sorted_route.end());
            uint64_t hash = HashRoute(sorted_route);

            //check if already exists
            if (route_hashes_.count(hash) > 0) continue;

            double cost = CalculateRouteCost(route, evaluator);

            // calculate route load
            int route_load = 0;
            for (int customer_id : route) {
                route_load += static_cast<int>(evaluator.GetDemand(customer_id));
            }

            // efficiency score to validate route quality
            double route_size = static_cast<double>(route.size());
            double load_ratio = static_cast<double>(route_load) / capacity;
            if (load_ratio < 0.01) load_ratio = 0.01;
            double efficiency = cost / (std::pow(route_size, 1.2) * std::pow(load_ratio, 0.5));


            CachedRoute cached;
            cached.nodes = route;
            cached.cost = cost;
            cached.efficiency = efficiency;
            cached.hash = hash;

            // generate bitmask
            cached.bitmask.assign(mask_words, 0);
            for (int cid : route) {
                if (cid < 2) continue;
                int idx = cid - 2;
                if (idx < num_clients) {
                    cached.bitmask[idx / 64] |= (1ULL << (idx % 64));
                }
            }

            routes_.push_back(std::move(cached));
            route_hashes_.insert(hash);
            total_routes_added_++;
        }

        //evict if over capacity
        if (routes_.size() > Config::ROUTE_POOL_MAX_SIZE) {
            EvictWorstRoutes();
        }
    }

    void RoutePool::EvictWorstRoutes() {
        std::sort(routes_.begin(), routes_.end());
        size_t target_size = static_cast<size_t>(Config::ROUTE_POOL_MAX_SIZE * 0.9);

        //rebuild hash set from survivors
        route_hashes_.clear();
        routes_.resize(target_size);
        for (const auto& r : routes_) {
            route_hashes_.insert(r.hash);
        }
    }

    void RoutePool::Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        routes_.clear();
        route_hashes_.clear();
        total_routes_added_ = 0;
    }

    size_t RoutePool::GetSize() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return routes_.size();
    }

    // beam search node for exploration
    struct BeamNode {
        double total_cost = 0.0;
        std::vector<int> route_indices;
        std::vector<uint64_t> mask;
        int last_route_idx = -1;
        int customers_covered = 0;
    };

    Individual RoutePool::SolveBeamSearch(ThreadSafeEvaluator* evaluator, Split& split, int beam_width) {

        std::vector<CachedRoute> sorted_routes;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (routes_.empty()) return Individual();
            sorted_routes = routes_;
        }

        int num_customers = evaluator->GetSolutionSize();
        int num_groups = evaluator->GetNumGroups();
        size_t mask_words = (num_customers + 63) / 64;
        const auto& perm = evaluator->GetPermutation();
        int capacity = evaluator->GetCapacity();

        std::sort(sorted_routes.begin(), sorted_routes.end());


        if (sorted_routes.size() > (size_t)Config::FRANKENSTEIN_MAX_SOURCE_ROUTES) {
            sorted_routes.resize((size_t)Config::FRANKENSTEIN_MAX_SOURCE_ROUTES);
        }

        // initialize beam
        std::vector<BeamNode> current_beam;
        BeamNode root;
        root.mask.assign(mask_words, 0);
        current_beam.push_back(root);

        int max_depth = num_groups + 2;
        int expansion_limit = 500;

        // beam search loop
        for (int depth = 0; depth < max_depth; ++depth) {
            std::vector<BeamNode> next_beam;
            bool any_expansion = false;

            for (const auto& node : current_beam) {
                if (node.customers_covered >= num_customers) {
                    next_beam.push_back(node);
                    continue;
                }

                int checked_count = 0;
                int added_count = 0;
                int current_branching = (node.route_indices.empty()) ? 100 : 10;
                int current_limit = (node.route_indices.empty()) ? 2000 : expansion_limit;

                for (int i = node.last_route_idx + 1; i < (int)sorted_routes.size(); ++i) {
                    if (checked_count++ > current_limit) break;

                    const auto& candidate = sorted_routes[i];
                    if (candidate.bitmask.size() != mask_words) continue;

                    // bitmask!!! ultra fast
                    bool overlap = false;
                    for (size_t mw = 0; mw < mask_words; ++mw) {
                        if (node.mask[mw] & candidate.bitmask[mw]) {
                            overlap = true;
                            break;
                        }
                    }
                    if (overlap) continue;

                    // create new node
                    BeamNode new_node = node;
                    new_node.last_route_idx = i;
                    new_node.route_indices.push_back(i);
                    new_node.total_cost += candidate.cost;

                    int route_customers = 0;
                    for (int cid : candidate.nodes) {
                        if (cid >= 2) route_customers++;
                    }
                    new_node.customers_covered += route_customers;

                    for (size_t mw = 0; mw < mask_words; ++mw) {
                        new_node.mask[mw] |= candidate.bitmask[mw];
                    }

                    next_beam.push_back(new_node);
                    added_count++;
                    any_expansion = true;

                    if (added_count >= current_branching) break;
                }
            }

            if (next_beam.empty()) break;

            //maximize coverage, minimize cost
            double avg_cost = 100.0;
            std::sort(next_beam.begin(), next_beam.end(),
                [avg_cost](const BeamNode& a, const BeamNode& b) {
                    double score_a = a.total_cost - (a.customers_covered * avg_cost * 2.0);
                    double score_b = b.total_cost - (b.customers_covered * avg_cost * 2.0);
                    return score_a < score_b;
                });

            if ((int)next_beam.size() > beam_width) {
                next_beam.resize(beam_width);
            }

            current_beam = std::move(next_beam);
            if (!any_expansion) break;
        }

        if (current_beam.empty()) return Individual();

        const BeamNode& best_node = current_beam[0];

        // construct result genotype
        std::vector<int> result_genotype(num_customers, -1);
        std::vector<int> result_permutation;
        result_permutation.reserve(num_customers);
        int current_group = 0;

        for (int r_idx : best_node.route_indices) {
            const auto& route = sorted_routes[r_idx];
            for (int cid : route.nodes) {
                if (cid < 2) continue;
                int gene_idx = cid - 2;
                if (gene_idx >= 0 && gene_idx < num_customers) {
                    result_genotype[gene_idx] = current_group;
                    result_permutation.push_back(cid);
                }
            }
            current_group++;
        }

        // handle leftovers
        std::vector<int> leftovers;
        for (int customer_id : perm) {
            if (customer_id < 2) continue;
            int gene_idx = customer_id - 2;
            if (gene_idx >= 0 && gene_idx < num_customers) {
                bool assigned = (best_node.mask[gene_idx / 64] & (1ULL << (gene_idx % 64)));
                if (!assigned) {
                    leftovers.push_back(customer_id);
                }
            }
        }

        if (!leftovers.empty()) {

            std::vector<int> group_loads(num_groups, 0);
            for (int i = 0; i < num_customers; ++i) {
                if (result_genotype[i] >= 0 && result_genotype[i] < num_groups) {
                    int customer_id = i + 2;
                    group_loads[result_genotype[i]] += static_cast<int>(evaluator->GetDemand(customer_id));
                }
            }


            std::sort(leftovers.begin(), leftovers.end(),
                [&](int a, int b) { return evaluator->GetDemand(a) > evaluator->GetDemand(b); });


            for (int customer_id : leftovers) {
                int gene_idx = customer_id - 2;
                if (gene_idx < 0 || gene_idx >= num_customers) continue;

                int demand = static_cast<int>(evaluator->GetDemand(customer_id));
                int best_group = -1;
                int min_slack = std::numeric_limits<int>::max();

                for (int g = 0; g < num_groups; ++g) {
                    int slack = capacity - group_loads[g];
                    if (slack >= demand && slack < min_slack) {
                        min_slack = slack;
                        best_group = g;
                    }
                }

                //fallback
                if (best_group == -1) {
                    int max_slack = -1;
                    for (int g = 0; g < num_groups; ++g) {
                        int slack = capacity - group_loads[g];
                        if (slack > max_slack) {
                            max_slack = slack;
                            best_group = g;
                        }
                    }
                }

                if (best_group == -1) {
                    best_group = (current_group < num_groups) ? current_group : 0;
                }

                result_genotype[gene_idx] = best_group;
                group_loads[best_group] += demand;
                result_permutation.push_back(customer_id);
            }
        }

        // validate genotype
        for (int i = 0; i < num_customers; ++i) {
            if (result_genotype[i] < 0) result_genotype[i] = 0;
        }

        Individual frankenstein(result_genotype);
        double fitness = evaluator->Evaluate(frankenstein.GetGenotype());
        frankenstein.SetFitness(fitness);

        return frankenstein;
    }

    std::vector<CachedRoute> RoutePool::GetBestRoutes(int n) const {
        std::lock_guard<std::mutex> lock(const_cast<RoutePool*>(this)->mutex_);

        if (routes_.empty()) return {};

        std::vector<CachedRoute> copy = routes_;
        std::sort(copy.begin(), copy.end());

        int count = std::min((int)copy.size(), n);
        copy.resize(count);
        return copy;
    }

    void RoutePool::ImportRoutes(const std::vector<CachedRoute>& imported_routes) {
        std::lock_guard<std::mutex> lock(mutex_);

        for (const auto& r : imported_routes) {

            if (route_hashes_.count(r.hash) > 0) continue;

            routes_.push_back(r);
            route_hashes_.insert(r.hash);
            total_routes_added_++;
        }

        if (routes_.size() > Config::ROUTE_POOL_MAX_SIZE) {
            std::sort(routes_.begin(), routes_.end());
            routes_.resize(Config::ROUTE_POOL_MAX_SIZE);

            route_hashes_.clear();
            for (const auto& r : routes_) {
                route_hashes_.insert(r.hash);
            }
        }
    }

}
