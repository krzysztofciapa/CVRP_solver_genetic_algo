#include "Split.hpp"
#include "Constants.hpp"
#include "ProblemGeometry.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <deque>
#include <iostream>
#include <limits>
#include <vector>

using namespace LcVRPContest;

Split::Split(const ThreadSafeEvaluator* evaluator) : evaluator_(evaluator) {
    capacity_ = evaluator_->GetCapacity();
    depot_idx_ = 0;
    num_customers_ = evaluator_->GetSolutionSize();
    int size = num_customers_ + 100;
    ResizeStructures(size);

    votes_buffer_.resize(evaluator_->GetNumGroups(), 0);
    segments_buffer_.reserve(size);
}

void Split::ResizeStructures(int size) {
    if ((int)D_.size() < size) {
        D_.resize(size);
        Q_.resize(size);
        V_.resize(size);
        pred_.resize(size);
        if ((int)in_window_.size() < size)
            in_window_.resize(size);
    }
}

void Split::ApplyMicroSplit(Individual& indiv, int start_idx, int end_idx, const ProblemGeometry* geometry, std::mt19937& rng) {
    //uses linear split with smart segment assignment to groups
    std::vector<int>& genes = indiv.AccessGenotype();
    const std::vector<int>& giant_tour = evaluator_->GetPermutation();
    int gt_size = static_cast<int>(giant_tour.size());
    int num_groups = evaluator_->GetNumGroups();
    int capacity = evaluator_->GetCapacity();

    if (start_idx < 0 || end_idx >= gt_size || start_idx > end_idx)
        return;

    int count = end_idx - start_idx + 1;

    ResizeStructures(count + 5);

    D_[0] = 0.0;
    Q_[0] = 0;
    V_[0] = 0.0;


    for (int i = 1; i <= count; ++i) {
        int curr_node = giant_tour[start_idx + i - 1];
        int curr_idx = (curr_node > 1) ? curr_node - 1 : 0;

        double dist = 0.0;
        if (i > 1) {
            int prev_node = giant_tour[start_idx + i - 2];
            int prev_idx = (prev_node > 1) ? prev_node - 1 : 0;
            dist = evaluator_->GetDist(prev_idx, curr_idx);
        }

        D_[i] = D_[i - 1] + dist;
        Q_[i] = Q_[i - 1] + evaluator_->GetDemand(curr_node);
    }


    dq_.clear();
    dq_.push_back(0);

    for (int i = 1; i <= count; ++i) {
        int curr_node = giant_tour[start_idx + i - 1];
        int curr_idx = (curr_node > 1) ? curr_node - 1 : 0;


        while (!dq_.empty() && (Q_[i] - Q_[dq_.front()] > capacity)) {
            dq_.pop_front();
        }


        if (dq_.empty()) {
            V_[i] = 1e30;
            continue;
        }

        int best = dq_.front();

        int start_node =
            giant_tour[start_idx + best];
        int s_idx = (start_node > 1) ? start_node - 1 : 0;

        double d_in = evaluator_->GetDist(depot_idx_, s_idx);
        double d_out = evaluator_->GetDist(curr_idx, depot_idx_);
        double d_route =
            D_[i] - D_[best + 1];

        V_[i] = V_[best] + d_in + d_route + d_out + Config::SPLIT_ROUTE_PENALTY;
        pred_[i] = best;


        if (i < count) {
            int next_node = giant_tour[start_idx + i];
            int next_idx = (next_node > 1) ? next_node - 1 : 0;
            double next_in = evaluator_->GetDist(depot_idx_, next_idx);

            double val_i = V_[i] - D_[i + 1] + next_in;

            while (!dq_.empty()) {
                int back = dq_.back();

                int back_next = giant_tour[start_idx + back];
                int back_idx = (back_next > 1) ? back_next - 1 : 0;
                double back_in = evaluator_->GetDist(depot_idx_, back_idx);

                double val_back = V_[back] - D_[back + 1] + back_in;

                if (val_i <= val_back) {
                    dq_.pop_back();
                }
                else {
                    break;
                }
            }
            dq_.push_back(i);
        }
    }

    if (V_[count] >= 1e29)
        return;

    segments_buffer_.clear();
    int curr = count;
    while (curr > 0) {
        int prev = pred_[curr];
        segments_buffer_.push_back({ prev + 1, curr, (double)(Q_[curr] - Q_[prev]) });
        curr = prev;
    }



    std::vector<int> last_customer_in_group(num_groups, 0);
    std::vector<int> group_loads(num_groups, 0);


    int initialized_count = 0;
    for (int k = start_idx - 1; k >= 0; --k) {
        if (initialized_count >= num_groups) break;
        int c_id = giant_tour[k];
        int gene_idx = c_id - 2;
        if (gene_idx >= 0 && gene_idx < (int)genes.size()) {
            int g = genes[gene_idx];
            if (g >= 0 && g < num_groups) {
                if (last_customer_in_group[g] == 0) {
                    last_customer_in_group[g] = c_id;
                    initialized_count++;
                }
            }
        }
    }

    int total_clients = (int)genes.size();
    std::vector<bool> in_window(total_clients, false);
    for (int k = 1; k <= count; ++k) {
        int c_id = giant_tour[start_idx + k - 1];
        int gene_idx = c_id - 2;
        if (gene_idx >= 0 && gene_idx < total_clients) {
            in_window[gene_idx] = true;
        }
    }

    for (int i = 0; i < total_clients; ++i) {
        if (in_window[i]) continue;
        int g = genes[i];
        if (g >= 0 && g < num_groups) {
            group_loads[g] += evaluator_->GetDemand(i + 2);
        }
    }


    for (int i = (int)segments_buffer_.size() - 1; i >= 0; --i) {
        const auto& seg = segments_buffer_[i];
        double seg_demand = seg.demand;
        int start_node_k = seg.start_k;
        int end_node_k = seg.end_k;

        int seg_first_client = giant_tour[start_idx + start_node_k - 1];
        int seg_last_client = giant_tour[start_idx + end_node_k - 1];

        int seg_first_idx = (seg_first_client > 1) ? seg_first_client - 1 : 0;

        int best_g = -1;
        double best_cost = 1e30;


        std::vector<int> candidates(num_groups);
        std::iota(candidates.begin(), candidates.end(), 0);

        for (int g : candidates) {
            if (group_loads[g] + (int)seg_demand <= capacity) {

                int last_node_id = last_customer_in_group[g];
                int last_idx = (last_node_id > 1) ? last_node_id - 1 : 0;


                double connection_cost = evaluator_->GetDist(last_idx, seg_first_idx);


                if (connection_cost < best_cost) {
                    best_cost = connection_cost;
                    best_g = g;
                }
            }
        }


        if (best_g == -1) {
            // pick group with most slack
            int max_val = -1;
            for (int g = 0; g < num_groups; ++g) {
                int rem = capacity - group_loads[g];
                if (rem > max_val) {
                    max_val = rem;
                    best_g = g;
                }
            }
        }

        group_loads[best_g] += (int)seg_demand;
        last_customer_in_group[best_g] = seg_last_client;

        for (int k = seg.start_k; k <= seg.end_k; ++k) {
            int c_id = giant_tour[start_idx + k - 1];
            int gene_idx = c_id - 2;
            if (gene_idx >= 0 && gene_idx < total_clients) {
                genes[gene_idx] = best_g;
            }
        }
    }
}

SplitResult Split::RunLinear(const std::vector<int>& giant_tour) {
    // uses linear split with O(n) complexity proposed by Vidal et al.
    PrecomputeStructures(giant_tour);
    int n = static_cast<int>(giant_tour.size());
    V_[0] = 0.0;

    std::deque<int> dq;
    dq.push_back(0);

    for (int i = 1; i <= n; ++i) {
        int curr_id = giant_tour[i - 1];
        int curr_idx = (curr_id > 1) ? curr_id - 1 : 0;

        while (!dq.empty()) {
            int front = dq.front();

            if (Q_[i] - Q_[front] > capacity_) {
                dq.pop_front();
                continue;
            }
            break;
        }

        if (dq.empty()) {
            V_[i] = std::numeric_limits<double>::max();
            continue;
        }

        int best_pred = dq.front();

        int start_node_id = giant_tour[best_pred];
        int start_idx = (start_node_id > 1) ? start_node_id - 1 : 0;

        double d_depot_start = evaluator_->GetDist(depot_idx_, start_idx);
        double d_end_depot = evaluator_->GetDist(curr_idx, depot_idx_);
        double d_internal = D_[i] - D_[best_pred + 1];

        V_[i] = V_[best_pred] + d_depot_start + d_internal + d_end_depot +
            Config::SPLIT_ROUTE_PENALTY;
        pred_[i] = best_pred;

        if (i < n) {
            if (V_[i] >= std::numeric_limits<double>::max())
                continue;

            int next_node_id = giant_tour[i];
            int next_idx = (next_node_id > 1) ? next_node_id - 1 : 0;

            double d_depot_next = evaluator_->GetDist(depot_idx_, next_idx);
            double val_i = V_[i] + d_depot_next - D_[i + 1];

            while (!dq.empty()) {
                int back = dq.back();
                int back_next_id = giant_tour[back];
                int back_next_idx = (back_next_id > 1) ? back_next_id - 1 : 0;

                double d_depot_back_next =
                    evaluator_->GetDist(depot_idx_, back_next_idx);
                double val_back = V_[back] + d_depot_back_next - D_[back + 1];

                if (val_i <= val_back) {
                    dq.pop_back();
                }
                else {
                    break;
                }
            }
            dq.push_back(i);
        }
    }

    return ReconstructResult(giant_tour);
}

SplitResult Split::RunBellman(const std::vector<int>& giant_tour) {
    //also works, but worse complexity
    PrecomputeStructures(giant_tour);
    int n = static_cast<int>(giant_tour.size());

    for (int i = 0; i <= n; ++i)
        V_[i] = 1e30;
    V_[0] = 0.0;

    for (int i = 1; i <= n; ++i) {
        int curr_id = giant_tour[i - 1];
        int curr_idx = curr_id - 1;

        for (int j = i - 1; j >= 0; --j) {
            int load = Q_[i] - Q_[j];
            if (load > capacity_)
                break;

            int start_node_id = giant_tour[j];
            int start_node_idx = start_node_id - 1;

            double d_depot_start = evaluator_->GetDist(depot_idx_, start_node_idx);
            double d_end_depot = evaluator_->GetDist(curr_idx, depot_idx_);
            double d_internal = D_[i] - D_[j + 1];

            double route_cost = d_depot_start + d_internal + d_end_depot;

            if (V_[j] + route_cost + Config::SPLIT_ROUTE_PENALTY < V_[i]) {
                V_[i] = V_[j] + route_cost + Config::SPLIT_ROUTE_PENALTY;
                pred_[i] = j;
            }
        }
    }
    return ReconstructResult(giant_tour);
}

SplitResult Split::ReconstructResult(const std::vector<int>& giant_tour) {
    SplitResult res;
    int n = static_cast<int>(giant_tour.size());

    if (V_[n] >= std::numeric_limits<double>::max()) {
        res.feasible = false;
        res.total_cost = std::numeric_limits<double>::max();
        return res;
    }

    res.feasible = true;
    res.total_cost = V_[n];

    int curr = n;
    while (curr > 0) {
        int prev = pred_[curr];
        std::vector<int> route;
        route.reserve(curr - prev);
        for (int k = prev + 1; k <= curr; ++k) {
            route.push_back(giant_tour[k - 1]);
        }
        res.optimized_routes.push_back(route);
        curr = prev;
    }
    std::reverse(res.optimized_routes.begin(), res.optimized_routes.end());

    res.group_assignment.assign(num_customers_, 0);

    for (size_t r_idx = 0; r_idx < res.optimized_routes.size(); ++r_idx) {
        for (int cust_id : res.optimized_routes[r_idx]) {
            int gene_idx = cust_id - 2;
            if (gene_idx >= 0 && gene_idx < num_customers_) {
                res.group_assignment[gene_idx] = static_cast<int>(r_idx);
            }
        }
    }
    return res;
}

void Split::PrecomputeStructures(const std::vector<int>& giant_tour) {
    int count = static_cast<int>(giant_tour.size());

    ResizeStructures(count + 1);

    D_[0] = 0.0;
    Q_[0] = 0;
    V_[0] = 0.0;

    for (int i = 1; i <= count; ++i) {
        int curr_node = giant_tour[i - 1];

        int curr_idx = (curr_node > 1) ? curr_node - 1 : 0;

        double dist = 0.0;
        if (i > 1) {
            int prev_node = giant_tour[i - 2];
            int prev_idx = (prev_node > 1) ? prev_node - 1 : 0;
            dist = evaluator_->GetDist(prev_idx, curr_idx);
        }

        D_[i] = D_[i - 1] + dist;
        Q_[i] = Q_[i - 1] + evaluator_->GetDemand(curr_node);
    }
}
