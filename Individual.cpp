#include "Individual.hpp"
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>

using namespace LcVRPContest;

using namespace std;

namespace LcVRPContest {

    Individual::Individual() : Individual(0) {}

    Individual::Individual(int size)
        : genotype_(size, 0),

        fitness_(numeric_limits<double>::max()), diversity_score_(0.0),
        biased_fitness_(numeric_limits<double>::max()), is_evaluated_(false)

    {
    }

    Individual::Individual(const vector<int>& genotype)
        : genotype_(genotype), fitness_(numeric_limits<double>::max()),
        diversity_score_(0.0), biased_fitness_(numeric_limits<double>::max()),
        is_evaluated_(false)

    {
    }

    Individual::Individual(vector<int>&& genotype)
        : genotype_(std::move(genotype)), fitness_(numeric_limits<double>::max()),
        diversity_score_(0.0), biased_fitness_(numeric_limits<double>::max()),
        is_evaluated_(false) {
    }

    const vector<int>& Individual::GetGenotype() const { return genotype_; }
    double Individual::GetFitness() const { return fitness_; }
    double Individual::GetDiversityScore() const { return diversity_score_; }
    double Individual::GetBiasedFitness() const { return biased_fitness_; }
    bool Individual::IsEvaluated() const { return is_evaluated_; }

    vector<int>& Individual::AccessGenotype() { return genotype_; }

    void Individual::SetFitness(double fitness) {
        fitness_ = fitness;
        is_evaluated_ = true;
    }

    void Individual::SetReturnCount(int count) { return_count_ = count; }
    int Individual::GetReturnCount() const { return return_count_; }

    void Individual::SetDiversityScore(double score) { diversity_score_ = score; }
    void Individual::SetBiasedFitness(double biased_fitness) {
        biased_fitness_ = biased_fitness;
    }

    bool Individual::operator<(const Individual& other) const {
        return GetFitness() < other.GetFitness();
    }
}

void Individual::PrintGroups(const std::vector<int>& permutation) const {
    int max_group = 0;
    for (int g : genotype_) {
        if (g > max_group)
            max_group = g;
    }
    int num_groups = max_group + 1;

    std::vector<std::vector<int>> routes(num_groups);
    for (int customer_id : permutation) {
        int sol_idx = customer_id - 2;

        if (sol_idx >= 0 && sol_idx < genotype_.size()) {
            int group_id = genotype_[sol_idx];
            if (group_id >= 0 && group_id < num_groups) {
                routes[group_id].push_back(customer_id - 1);
            }
        }
    }
    int route_counter = 1;
    for (int i = 0; i < num_groups; ++i) {
        if (routes[i].empty())
            continue;

        std::cout << "Route #" << route_counter++ << ":";
        for (int customer : routes[i]) {
            std::cout << " " << customer;
        }
        std::cout << std::endl;
    }
}
void Individual::Canonicalize() {

    static thread_local std::vector<int> mapping_buffer;

    int max_id = 0;
    for (int g : genotype_) {
        if (g > max_id)
            max_id = g;
    }

    if ((int)mapping_buffer.size() <= max_id) {
        mapping_buffer.resize(max_id + 1, -1);
    }
    else {
        std::fill(mapping_buffer.begin(), mapping_buffer.begin() + max_id + 1, -1);
    }

    int next_code = 0;

    for (int& gene : genotype_) {

        if (gene < 0)
            continue;

        if (mapping_buffer[gene] == -1) {
            mapping_buffer[gene] = next_code++;
        }

        gene = mapping_buffer[gene];
    }
}
