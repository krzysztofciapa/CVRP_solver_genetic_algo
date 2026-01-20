#include "Mutator.hpp"
#include "AdaptiveOperators.hpp"
#include "Constants.hpp"
#include <cmath>
#include <limits>
#include <numeric>

using namespace LcVRPContest;

Mutator::Mutator()
    : evaluator_(nullptr), geometry_(nullptr), split_ptr_(nullptr) {
}

void Mutator::Initialize(ThreadSafeEvaluator* eval, const ProblemGeometry* geo,
    Split* split) {
    evaluator_ = eval;
    geometry_ = geo;
    split_ptr_ = split;

    int num_clients = eval->GetSolutionSize();
    int num_groups = eval->GetNumGroups();

    //buffers
    removed_indices_buffer_.reserve(num_clients);
    is_removed_buffer_.resize(num_clients, false);
    group_votes_buffer_.resize(num_groups, 0);
    candidates_buffer_.reserve(20);


    group_loads_buffer_.resize(num_groups);
    group_clients_buffer_.resize(num_groups);
    for (auto& vec : group_clients_buffer_) vec.reserve(num_clients / num_groups + 10);
    group_overload_buffer_.reserve(num_groups);
    relatedness_buffer_.reserve(num_clients);

    routes_buffer_.resize(num_groups);
    for (auto& r : routes_buffer_) r.reserve(num_clients / num_groups + 20);
    overflow_groups_buffer_.reserve(num_groups);
    check_indices_buffer_.resize(num_clients);
    temp_route_buffer_.reserve(num_clients / num_groups + 20);


    int dim = eval->GetDimension();
    customer_ranks_.resize(dim + 1, 0);
    const auto& perm = eval->GetPermutation();
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] >= 0 && perm[i] < (int)customer_ranks_.size()) {
            customer_ranks_[perm[i]] = static_cast<int>(i);
        }
    }
}

bool Mutator::ApplyRuinRecreate(Individual& indiv, double intensity, bool is_exploitation, std::mt19937& rng) {
    if (!geometry_ || !geometry_->HasNeighbors())
        return false;

    std::vector<int>& genes = indiv.AccessGenotype();
    int num_clients = static_cast<int>(genes.size());
    int num_groups = evaluator_->GetNumGroups();

    int center_idx = rng() % num_clients;

    int min_rem = Config::RUIN_MIN_REMOVED;
    double pct;
    if (is_exploitation) {
        // exploitation: smaller windows 10-25%
        pct = Config::RUIN_BASE_PCT_EXPLOITATION +
            (Config::RUIN_INTENSITY_SCALE_EXPLOITATION * intensity);
    }
    else {
        // exploration: larger windows 30-70%
        pct = Config::RUIN_BASE_PCT + (Config::RUIN_INTENSITY_SCALE * intensity);
    }

    int max_rem = std::max(min_rem + 5, (int)(num_clients * pct));
    if (max_rem >= num_clients)
        max_rem = num_clients - 1;


    const auto& neighbors = geometry_->GetNeighbors(center_idx);
    int available_neighbors = static_cast<int>(neighbors.size());

    if (max_rem > available_neighbors)
        max_rem = available_neighbors;
    int range = max_rem - min_rem;
    if (range <= 0)
        range = 1;

    int num_removed = min_rem + (rng() % range);
    if (num_removed < 1)
        return false;

    removed_indices_buffer_.clear();
    removed_indices_buffer_.push_back(center_idx);

    for (int i = 0; i < num_removed && i < available_neighbors; ++i) {
        removed_indices_buffer_.push_back(neighbors[i]);
    }


    std::fill(is_removed_buffer_.begin(), is_removed_buffer_.end(), false);

    for (int idx : removed_indices_buffer_) {
        if (idx < num_clients)
            is_removed_buffer_[idx] = true;
    }

    for (int idx : removed_indices_buffer_)
        genes[idx] = -1;

    std::shuffle(removed_indices_buffer_.begin(), removed_indices_buffer_.end(),
        rng);


    int capacity = evaluator_->GetCapacity();
    if (group_loads_buffer_.size() != num_groups) group_loads_buffer_.resize(num_groups);
    std::fill(group_loads_buffer_.begin(), group_loads_buffer_.end(), 0);

    for (int i = 0; i < num_clients; ++i) {
        if (!is_removed_buffer_[i]) {
            int g = genes[i];
            if (g >= 0 && g < num_groups) {
                group_loads_buffer_[g] += evaluator_->GetDemand(i + 2);
            }
        }
    }

    // neighbour voting and capacity respect
    for (int client_idx : removed_indices_buffer_) {

        std::fill(group_votes_buffer_.begin(), group_votes_buffer_.end(), 0);


        const auto& client_neighbors = geometry_->GetNeighbors(client_idx);

        for (int neighbor : client_neighbors) {
            if (neighbor >= num_clients)
                continue;
            if (is_removed_buffer_[neighbor])
                continue;
            int neighbor_group = genes[neighbor];
            if (neighbor_group >= 0 && neighbor_group < num_groups) {
                group_votes_buffer_[neighbor_group]++;
            }
        }

        int best_g = -1;
        int max_votes = -1;

        int demand = evaluator_->GetDemand(client_idx + 2);


        for (int g = 0; g < num_groups; ++g) {
            if (group_loads_buffer_[g] + demand <= capacity) {
                if (group_votes_buffer_[g] > max_votes) {
                    max_votes = group_votes_buffer_[g];
                    best_g = g;
                }
            }
        }

        // fallback, ignore capacity
        if (best_g == -1) {
            max_votes = -1;
            for (int g = 0; g < num_groups; ++g) {
                if (group_votes_buffer_[g] > max_votes) {
                    max_votes = group_votes_buffer_[g];
                    best_g = g;
                }
            }
        }

        if (best_g == -1) {
            best_g = rng() % num_groups;
        }

        genes[client_idx] = best_g;
        group_loads_buffer_[best_g] += demand;
        is_removed_buffer_[client_idx] = false;
    }

    return true;
}


bool Mutator::ApplyAdaptiveLNS(Individual& indiv, LNSStrategy strategy, double intensity, bool is_exploitation, std::mt19937& rng) {
    // in principle similar to ruin-recreate, but different removal strategies, same recreate
    if (!geometry_ || !geometry_->HasNeighbors())
        return false;

    std::vector<int>& genes = indiv.AccessGenotype();
    int num_clients = static_cast<int>(genes.size());
    int num_groups = evaluator_->GetNumGroups();
    int capacity = evaluator_->GetCapacity();
    const auto& permutation = evaluator_->GetPermutation();

    // calculate ruin size based on intensity and island type
    int min_rem = Config::RUIN_MIN_REMOVED;
    double pct;
    if (is_exploitation) {
        pct = Config::RUIN_BASE_PCT_EXPLOITATION +
            (Config::RUIN_INTENSITY_SCALE_EXPLOITATION * intensity);
    }
    else {
        pct = Config::RUIN_BASE_PCT + (Config::RUIN_INTENSITY_SCALE * intensity);
    }
    int target_removed = std::max(min_rem, (int)(num_clients * pct));
    if (target_removed >= num_clients)
        target_removed = num_clients / 2;


    removed_indices_buffer_.clear();
    std::fill(is_removed_buffer_.begin(), is_removed_buffer_.end(), false);

    switch (strategy) {
    case LNSStrategy::RANDOM_CLUSTER: {
        //random center + geometric neighbors - classic RR
        int center_idx = rng() % num_clients;
        removed_indices_buffer_.push_back(center_idx);

        const auto& neighbors = geometry_->GetNeighbors(center_idx);
        for (int i = 0; i < (int)neighbors.size() && (int)removed_indices_buffer_.size() < target_removed; ++i) {
            if (neighbors[i] < num_clients) {
                removed_indices_buffer_.push_back(neighbors[i]);
            }
        }
        break;
    }

    case LNSStrategy::WORST_ROUTES: {
        //destroy bad routes
        if (group_loads_buffer_.size() != num_groups) group_loads_buffer_.resize(num_groups);
        std::fill(group_loads_buffer_.begin(), group_loads_buffer_.end(), 0);

        if (group_clients_buffer_.size() != num_groups) group_clients_buffer_.resize(num_groups);
        for (auto& vec : group_clients_buffer_) vec.clear();

        for (int i = 0; i < num_clients; ++i) {
            int g = genes[i];
            if (g >= 0 && g < num_groups) {
                int demand = evaluator_->GetDemand(i + 2);
                group_loads_buffer_[g] += demand;
                group_clients_buffer_[g].push_back(i);
            }
        }

        group_overload_buffer_.clear();
        for (int g = 0; g < num_groups; ++g) {
            if (group_loads_buffer_[g] > capacity) {
                group_overload_buffer_.push_back({ group_loads_buffer_[g] - capacity, g });
            }
        }
        std::sort(group_overload_buffer_.rbegin(), group_overload_buffer_.rend());

        for (size_t i = 0; i < group_overload_buffer_.size() && (int)removed_indices_buffer_.size() < target_removed; ++i) {
            int g = group_overload_buffer_[i].second;
            const auto& clients = group_clients_buffer_[g];
            for (size_t c = 0; c < clients.size() && (int)removed_indices_buffer_.size() < target_removed; ++c) {
                removed_indices_buffer_.push_back(clients[c]);
            }
        }

        while ((int)removed_indices_buffer_.size() < target_removed) {
            int idx = rng() % num_clients;
            bool already_in = false;
            for (int r : removed_indices_buffer_) {
                if (r == idx) { already_in = true; break; }
            }
            if (!already_in) removed_indices_buffer_.push_back(idx);
        }
        break;
    }

    case LNSStrategy::RELATED_REMOVAL: {
        // remove clients with similard demand and / or proximity
        int seed_idx = rng() % num_clients;
        int seed_demand = evaluator_->GetDemand(seed_idx + 2);
        removed_indices_buffer_.push_back(seed_idx);


        relatedness_buffer_.clear();
        const auto& seed_neighbors = geometry_->GetNeighbors(seed_idx);

        for (int i = 0; i < num_clients; ++i) {
            if (i == seed_idx) continue;

            int demand = evaluator_->GetDemand(i + 2);
            double demand_similarity = 1.0 / (1.0 + std::abs(demand - seed_demand));


            double proximity = 0.1;
            bool found_neighbor = false;
            for (int j = 0; j < (int)seed_neighbors.size() && j < 20 && !found_neighbor; ++j) {
                if (seed_neighbors[j] == i) {
                    proximity = 1.0 - (j / 20.0);
                    found_neighbor = true;
                }
            }

            double score = demand_similarity * 0.5 + proximity * 0.5;
            relatedness_buffer_.push_back({ score, i });
        }

        std::sort(relatedness_buffer_.rbegin(), relatedness_buffer_.rend());

        for (int i = 0; i < (int)relatedness_buffer_.size() && (int)removed_indices_buffer_.size() < target_removed; ++i) {
            removed_indices_buffer_.push_back(relatedness_buffer_[i].second);
        }
        break;
    }

    case LNSStrategy::PERMUTATION_BASED: {
        // remove a block from the global permutation
        int perm_size = static_cast<int>(permutation.size());
        if (perm_size < target_removed + 2) {

            int center_idx = rng() % num_clients;
            removed_indices_buffer_.push_back(center_idx);
            const auto& neighbors = geometry_->GetNeighbors(center_idx);
            for (int i = 0; i < (int)neighbors.size() && (int)removed_indices_buffer_.size() < target_removed; ++i) {
                if (neighbors[i] < num_clients) {
                    removed_indices_buffer_.push_back(neighbors[i]);
                }
            }
        }
        else {

            int start_pos = rng() % (perm_size - target_removed);
            for (int i = 0; i < target_removed; ++i) {
                int perm_id = permutation[start_pos + i];
                if (perm_id >= 2) {
                    int client_idx = perm_id - 2;
                    if (client_idx >= 0 && client_idx < num_clients) {
                        removed_indices_buffer_.push_back(client_idx);
                    }
                }
            }
        }
        break;
    }

    default:
        return false;
    }

    if (removed_indices_buffer_.empty())
        return false;

    for (int idx : removed_indices_buffer_) {
        if (idx >= 0 && idx < num_clients) {
            is_removed_buffer_[idx] = true;
            genes[idx] = -1;
        }
    }

    std::shuffle(removed_indices_buffer_.begin(), removed_indices_buffer_.end(), rng);

    //recreate phase
    if (group_loads_buffer_.size() != num_groups) group_loads_buffer_.resize(num_groups);
    std::fill(group_loads_buffer_.begin(), group_loads_buffer_.end(), 0);

    for (int i = 0; i < num_clients; ++i) {
        if (!is_removed_buffer_[i]) {
            int g = genes[i];
            if (g >= 0 && g < num_groups) {
                group_loads_buffer_[g] += evaluator_->GetDemand(i + 2);
            }
        }
    }

    for (int client_idx : removed_indices_buffer_) {
        if (client_idx < 0 || client_idx >= num_clients) continue;


        std::fill(group_votes_buffer_.begin(), group_votes_buffer_.end(), 0);


        const auto& client_neighbors = geometry_->GetNeighbors(client_idx);
        for (int neighbor : client_neighbors) {
            if (neighbor >= num_clients) continue;
            if (is_removed_buffer_[neighbor]) continue;
            int neighbor_group = genes[neighbor];
            if (neighbor_group >= 0 && neighbor_group < num_groups) {
                group_votes_buffer_[neighbor_group]++;
            }
        }

        //respect the neighbors in the global permutation too, bc they are "free"
        for (int p = 0; p < (int)permutation.size(); ++p) {
            if (permutation[p] - 2 == client_idx) {

                if (p > 0) {
                    int prev_client = permutation[p - 1] - 2;
                    if (prev_client >= 0 && prev_client < num_clients &&
                        !is_removed_buffer_[prev_client]) {
                        int prev_group = genes[prev_client];
                        if (prev_group >= 0 && prev_group < num_groups) {
                            group_votes_buffer_[prev_group] += 2;
                        }
                    }
                }
                if (p < (int)permutation.size() - 1) {
                    int next_client = permutation[p + 1] - 2;
                    if (next_client >= 0 && next_client < num_clients &&
                        !is_removed_buffer_[next_client]) {
                        int next_group = genes[next_client];
                        if (next_group >= 0 && next_group < num_groups) {
                            group_votes_buffer_[next_group] += 2;
                        }
                    }
                }
                break;
            }
        }

        int best_g = -1;
        int max_votes = -1;
        int demand = evaluator_->GetDemand(client_idx + 2);

        for (int g = 0; g < num_groups; ++g) {
            if (group_loads_buffer_[g] + demand <= capacity) {
                if (group_votes_buffer_[g] > max_votes) {
                    max_votes = group_votes_buffer_[g];
                    best_g = g;
                }
            }
        }

        if (best_g == -1) {
            max_votes = -1;
            for (int g = 0; g < num_groups; ++g) {
                if (group_votes_buffer_[g] > max_votes) {
                    max_votes = group_votes_buffer_[g];
                    best_g = g;
                }
            }
        }

        if (best_g == -1) {
            best_g = rng() % num_groups;
        }

        genes[client_idx] = best_g;
        group_loads_buffer_[best_g] += demand;
        is_removed_buffer_[client_idx] = false;
    }

    return true;
}


bool Mutator::ApplySmartSpatialMove(Individual& indiv, std::mt19937& rng) {
    // moves a random client to a neighboring group based on geometry and permutation
    if (!geometry_)
        return false;

    std::vector<int>& genes = indiv.AccessGenotype();
    int size = static_cast<int>(genes.size());
    int num_groups = evaluator_->GetNumGroups();

    if (size == 0) return false;

    int client_idx = rng() % size;
    int current_group = genes[client_idx];
    int client_id = GetClientId(client_idx);

    candidates_buffer_.clear();

    int rank = 0;
    if (client_id >= 0 && client_id < (int)customer_ranks_.size()) {
        rank = customer_ranks_[client_id];
    }

    const auto& global_perm = evaluator_->GetPermutation();
    int perm_size = (int)global_perm.size();


    if (rank > 0 && rank < perm_size) {
        int prev_id = global_perm[rank - 1];
        if (prev_id >= 2) {
            int prev_idx = prev_id - 2;
            if (prev_idx >= 0 && prev_idx < size) {
                int g = genes[prev_idx];
                if (g != current_group && g >= 0 && g < num_groups) {
                    candidates_buffer_.push_back(g);
                    candidates_buffer_.push_back(g);
                    candidates_buffer_.push_back(g);
                }
            }
        }
    }

    if (rank < perm_size - 1 && rank >= 0) {
        int next_id = global_perm[rank + 1];
        if (next_id >= 2) {
            int next_idx = next_id - 2;
            if (next_idx >= 0 && next_idx < size) {
                int g = genes[next_idx];
                if (g != current_group && g >= 0 && g < num_groups) {

                    candidates_buffer_.push_back(g);
                    candidates_buffer_.push_back(g);
                    candidates_buffer_.push_back(g);
                }
            }
        }
    }
    //geometric neighbors
    const auto& geo_neighbors = geometry_->GetNeighbors(client_idx);
    int limit = std::min((int)geo_neighbors.size(), 15);

    for (int k = 0; k < limit; ++k) {
        int neighbor_idx = geo_neighbors[k];
        if (neighbor_idx >= size) continue;

        int g = genes[neighbor_idx];
        if (g != current_group && g >= 0 && g < num_groups) {
            candidates_buffer_.push_back(g);
        }
    }

    if (candidates_buffer_.empty()) {

        if (rng() % 100 < 5) {
            int random_g = rng() % num_groups;
            if (random_g != current_group) {
                genes[client_idx] = random_g;
                return true;
            }
        }
        return false;
    }

    //candidate selection
    int best_target_group = candidates_buffer_[rng() % candidates_buffer_.size()];

    if (best_target_group != -1) {
        genes[client_idx] = best_target_group;
        return true;
    }

    return false;
}

bool Mutator::ApplyMicroSplitMutation(Individual& child, double stagnation_factor, int level, std::mt19937& rng) {
    if (!split_ptr_)
        return false;

    int n_perm = (int)evaluator_->GetPermutation().size();
    if (n_perm < 10)
        return false;


    int base_min, base_max;
    switch (level) {
    case 0: // large windows
        base_min = 20;
        base_max = std::max(40, n_perm / 8);
        break;
    case 1:
        base_min = 15;
        base_max = std::max(30, n_perm / 10);
        break; // medium windows
    default:
        base_min = 8;
        base_max = std::max(15, n_perm / 15);
        break; //  small windows
    }

    // a little flexibility with stagnation factor
    int current_max = base_max + (int)((n_perm * 0.3) * stagnation_factor);
    int current_min = base_min + (int)(20 * stagnation_factor);

    current_max = std::min(current_max, n_perm - 1);
    current_min = std::min(current_min, current_max - 1);

    if (current_min < 5)
        current_min = 5;

    std::uniform_int_distribution<int> dist_len(current_min, current_max);
    int window_len = dist_len(rng);

    if (window_len >= n_perm)
        window_len = n_perm - 1;

    std::uniform_int_distribution<int> dist_start(0, n_perm - window_len);
    int start_idx = dist_start(rng);
    int end_idx = start_idx + window_len - 1;


    split_ptr_->ApplyMicroSplit(child, start_idx, end_idx, geometry_, rng);

    return true;
}

bool Mutator::AggressiveMutate(Individual& indiv, std::mt19937& rng) {
    std::vector<int>& genes = indiv.AccessGenotype();
    if (genes.empty())
        return false;

    int min_g = evaluator_->GetLowerBound();
    int max_g = evaluator_->GetUpperBound();
    int size = static_cast<int>(genes.size());

    std::uniform_int_distribution<int> dist_idx(0, size - 1);
    std::uniform_int_distribution<int> dist_grp(min_g, max_g);

    int start_idx = dist_idx(rng);
    std::uniform_int_distribution<int> dist_end(start_idx, size - 1);
    int end_idx = dist_end(rng);

    if (dist_idx(rng) % 2 == 0) {

        std::shuffle(genes.begin() + start_idx, genes.begin() + end_idx + 1, rng);
    }
    else {

        for (int i = start_idx; i <= end_idx; i++) {
            genes[i] = dist_grp(rng);
        }
    }
    return true;
}

bool Mutator::ApplySimpleMutation(Individual& indiv, std::mt19937& rng) {
    int min_g = evaluator_->GetLowerBound();
    int max_g = evaluator_->GetUpperBound();
    std::vector<int>& genes = indiv.AccessGenotype();
    if (genes.empty())
        return false;

    std::uniform_int_distribution<int> dist_idx(
        0, static_cast<int>(genes.size()) - 1);
    std::uniform_int_distribution<int> dist_grp(min_g, max_g);


    if (rng() % 2 == 0) {
        int idx1 = dist_idx(rng);
        int idx2 = dist_idx(rng);
        std::swap(genes[idx1], genes[idx2]);
    }
    else {
        int idx = dist_idx(rng);
        int grp = dist_grp(rng);
        while (grp == genes[idx] && min_g != max_g)
            grp = dist_grp(rng);
        genes[idx] = grp;
    }
    return true;
}


bool Mutator::ApplyReturnMinimizer(Individual& indiv, std::mt19937& rng) {
    // moves clients from overloaded groups to reduce returns
    if (!evaluator_)
        return false;

    std::vector<int>& genes = indiv.AccessGenotype();
    int num_clients = static_cast<int>(genes.size());
    int num_groups = evaluator_->GetNumGroups();
    int capacity = evaluator_->GetCapacity();
    const auto& permutation = evaluator_->GetPermutation();

    if (routes_buffer_.size() != num_groups) routes_buffer_.resize(num_groups);
    for (auto& r : routes_buffer_) r.clear();

    if (group_loads_buffer_.size() != num_groups) group_loads_buffer_.resize(num_groups);
    std::fill(group_loads_buffer_.begin(), group_loads_buffer_.end(), 0);

    for (int perm_id : permutation) {

        if (perm_id < 2) continue;

        int client_idx = perm_id - 2;
        if (client_idx < 0 || client_idx >= num_clients) continue;

        int g = genes[client_idx];
        if (g >= 0 && g < num_groups) {
            routes_buffer_[g].push_back(client_idx);
            group_loads_buffer_[g] += evaluator_->GetDemand(perm_id);
        }
    }

    overflow_groups_buffer_.clear();
    for (int g = 0; g < num_groups; ++g) {
        if (group_loads_buffer_[g] > capacity) {
            overflow_groups_buffer_.push_back(g);
        }
    }

    if (overflow_groups_buffer_.empty()) return false;


    int target_g_idx = rng() % overflow_groups_buffer_.size();
    int source_g = overflow_groups_buffer_[target_g_idx];
    const auto& route = routes_buffer_[source_g];

    if (route.empty()) return false;

    int best_victim_idx = -1;
    double max_saving = -1.0;

    double full_cost = evaluator_->GetRouteCost(EncodeRoute(route));

    int check_limit = (route.size() > 20) ? 20 : (int)route.size();

    check_indices_buffer_.resize(route.size());
    std::iota(check_indices_buffer_.begin(), check_indices_buffer_.end(), 0);
    if (route.size() > 20) {
        std::shuffle(check_indices_buffer_.begin(), check_indices_buffer_.end(), rng);
    }

    for (int k = 0; k < check_limit; ++k) {
        int idx_in_route = check_indices_buffer_[k];
        int client_idx = route[idx_in_route];

        temp_route_buffer_.clear();
        temp_route_buffer_.reserve(route.size() - 1);
        for (int c : route) {
            if (c != client_idx) temp_route_buffer_.push_back(GetClientId(c));
        }

        double cost_without = evaluator_->GetRouteCost(temp_route_buffer_);
        double saving = full_cost - cost_without;

        if (saving > max_saving) {
            max_saving = saving;
            best_victim_idx = client_idx;
        }
    }

    if (best_victim_idx == -1) return false;

    int victim_idx = best_victim_idx;
    int victim_id = GetClientId(victim_idx);
    int victim_demand = evaluator_->GetDemand(victim_id);

    int best_target = -1;


    if (geometry_) {
        const auto& neighbors = geometry_->GetNeighbors(victim_idx);
        bool target_found = false;
        for (size_t i = 0; i < neighbors.size() && !target_found; ++i) {
            int neighbor = neighbors[i];
            if (neighbor >= num_clients) continue;
            int neighbor_group = genes[neighbor];
            if (neighbor_group == source_g) continue;
            if (neighbor_group < 0 || neighbor_group >= num_groups) continue;

            if (group_loads_buffer_[neighbor_group] + victim_demand <= capacity) {
                best_target = neighbor_group;
                target_found = true;
            }
        }
    }

    if (best_target == -1) {
        int min_load = std::numeric_limits<int>::max();
        for (int g = 0; g < num_groups; ++g) {
            if (g == source_g) continue;
            if (group_loads_buffer_[g] < min_load) {
                min_load = group_loads_buffer_[g];
                best_target = g;
            }
        }
    }

    if (best_target != -1) {
        genes[victim_idx] = best_target;
        return true;
    }

    return false;
}

int Mutator::GetClientId(int idx) const { return idx + 2; }
std::vector<int> Mutator::EncodeRoute(const std::vector<int>& idx_route) const {
    std::vector<int> id_route;
    id_route.reserve(idx_route.size());
    for (int idx : idx_route) id_route.push_back(idx + 2);
    return id_route;
}

bool Mutator::ApplyMergeSplit(Individual& indiv, std::mt19937& rng) {
    // merge-split operator: merges two groups and redistributes customers to balance loads via split
    if (!evaluator_)
        return false;

    std::vector<int>& genes = indiv.AccessGenotype();
    int num_clients = static_cast<int>(genes.size());
    int num_groups = evaluator_->GetNumGroups();
    const auto& permutation = evaluator_->GetPermutation();
    int capacity = evaluator_->GetCapacity();

    if (num_groups < 2 || num_clients < 2)
        return false;

    std::vector<int> group_counts(num_groups, 0);
    for (int i = 0; i < num_clients; ++i) {
        int g = genes[i];
        if (g >= 0 && g < num_groups) {
            group_counts[g]++;
        }
    }

    std::vector<int> non_empty_groups;
    for (int g = 0; g < num_groups; ++g) {
        if (group_counts[g] > 0) {
            non_empty_groups.push_back(g);
        }
    }

    if (non_empty_groups.size() < 2)
        return false;

    int idx1 = rng() % non_empty_groups.size();
    int idx2;
    do {
        idx2 = rng() % non_empty_groups.size();
    } while (idx2 == idx1);

    int g1 = non_empty_groups[idx1];
    int g2 = non_empty_groups[idx2];


    std::vector<int> merged_giant_tour;
    merged_giant_tour.reserve(group_counts[g1] + group_counts[g2]);

    for (int perm_id : permutation) {
        if (perm_id < 2) continue;
        int client_idx = perm_id - 2;
        if (client_idx < 0 || client_idx >= num_clients) continue;

        int g = genes[client_idx];
        if (g == g1 || g == g2) {
            merged_giant_tour.push_back(perm_id);
        }
    }

    if (merged_giant_tour.size() < 2)
        return false;


    if (!split_ptr_) return false;
    SplitResult result = split_ptr_->RunLinear(merged_giant_tour);

    if (!result.feasible) return false;

    // we only allow up to 2 routes after split for merge-split
    if (result.optimized_routes.size() > 2) {
        return false;
    }

    for (int perm_id : merged_giant_tour) {
        genes[perm_id - 2] = -1;
    }

    if (!result.optimized_routes.empty()) {
        for (int c_id : result.optimized_routes[0]) {
            genes[c_id - 2] = g1;
        }
    }


    if (result.optimized_routes.size() > 1) {
        for (int c_id : result.optimized_routes[1]) {
            genes[c_id - 2] = g2;
        }
    }
    return true;
}

bool Mutator::EliminateReturns(Individual& indiv, std::mt19937& rng) {
    // identifies inefficient return trips and reassigns clients to eliminate them
    if (!evaluator_)
        return false;

    std::vector<int>& genes = indiv.AccessGenotype();
    int num_clients = static_cast<int>(genes.size());
    int num_groups = evaluator_->GetNumGroups();
    int capacity = evaluator_->GetCapacity();
    const auto& permutation = evaluator_->GetPermutation();


    std::vector<int> group_loads(num_groups, 0);
    std::vector<std::vector<int>> group_clients(num_groups);

    for (int perm_id : permutation) {
        if (perm_id < 2) continue;
        int client_idx = perm_id - 2;
        if (client_idx < 0 || client_idx >= num_clients) continue;
        int g = genes[client_idx];
        if (g >= 0 && g < num_groups) {
            group_clients[g].push_back(client_idx);
        }
    }

    std::vector<int> clients_to_move;
    bool found_inefficient = false;

    for (int g = 0; g < num_groups; ++g) {
        if (group_clients[g].empty()) continue;

        int current_load = 0;
        int full_trips = 0;
        std::vector<int> tail_clients;
        int total_load_check = 0;

        for (int client_idx : group_clients[g]) {
            int demand = evaluator_->GetDemand(client_idx + 2);
            total_load_check += demand;

            if (current_load + demand > capacity) {
                full_trips++;
                tail_clients.clear();

                current_load = 0;

            }
            current_load += demand;
            tail_clients.push_back(client_idx);
        }
        if (full_trips > 0 && current_load > 0) {

            double tail_efficiency = (double)current_load / capacity;

            if (tail_efficiency < 0.70) {

                for (int c : tail_clients) {
                    clients_to_move.push_back(c);
                    genes[c] = -1;
                    group_loads[g] = total_load_check - current_load;
                }
                found_inefficient = true;
            }
            else {

                group_loads[g] = total_load_check;
            }
        }
        else {

            group_loads[g] = total_load_check;
        }
    }

    if (clients_to_move.empty())
        return false;

    // sort clients by descending demand to place larger ones first
    std::sort(clients_to_move.begin(), clients_to_move.end(),
        [&](int a, int b) {
            return evaluator_->GetDemand(a + 2) > evaluator_->GetDemand(b + 2);
        });

    for (int client_idx : clients_to_move) {
        int demand = evaluator_->GetDemand(client_idx + 2);
        int best_g = -1;
        int min_slack = INT_MAX;

        //try to fit without creating new return trip
        for (int g = 0; g < num_groups; ++g) {


            int remainder = group_loads[g] % capacity;
            int space_in_last_trip = capacity - remainder;
            if (remainder == 0 && group_loads[g] > 0) space_in_last_trip = 0;

            if (space_in_last_trip >= demand) {
                int slack = space_in_last_trip - demand;
                if (slack < min_slack) {
                    min_slack = slack;
                    best_g = g;
                }
            }
        }


        if (best_g == -1) {
            int min_total = INT_MAX;
            for (int g = 0; g < num_groups; ++g) {
                if (group_loads[g] < min_total) {
                    min_total = group_loads[g];
                    best_g = g;
                }
            }
        }

        if (best_g != -1) {
            genes[client_idx] = best_g;
            group_loads[best_g] += demand;
        }
        else {

            genes[client_idx] = rng() % num_groups;
        }
    }

    return true;
}
