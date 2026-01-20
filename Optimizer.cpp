#include "Optimizer.hpp"
#include "Constants.hpp"
#include "ProblemData.hpp"
#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>

using namespace LcVRPContest;
using namespace std;

Optimizer::Optimizer(Evaluator& evaluator)
    : evaluator_(evaluator), rng_(random_device{}()),
    current_best_fitness_(numeric_limits<double>::max()), is_running_(false) {
    const ProblemData& data = evaluator_.GetProblemData();
    int num_groups = evaluator.GetNumGroups();


    unsigned int hw_threads = std::thread::hardware_concurrency();

    int threads_to_use = (int)hw_threads - 4;
    if (threads_to_use < 4) threads_to_use = 4;

    num_islands_ = threads_to_use;
    num_islands_ = 6; // 384 cores / 60 instances at once ~~6 threads per instance, the hardcoding is on purpose

    if (num_islands_ % 2 != 0) num_islands_--;


    evaluators_.resize(num_islands_);
    islands_.resize(num_islands_);


    island_generations_ = new std::atomic<long long>[num_islands_];

    for (int i = 0; i < num_islands_; ++i) {
        island_generations_[i] = 0;
    }

    int num_customers = data.GetNumCustomers();

    for (int i = 0; i < num_islands_; ++i) {
        evaluators_[i] = new ThreadSafeEvaluator(data, num_groups);


        int pop_size;
        if (num_customers > Config::HUGE_INSTANCE_THRESHOLD) {

            pop_size = (i % 2 == 0) ? 30 : 20;
        }
        else if (num_customers > Config::LARGE_INSTANCE_THRESHOLD) {

            pop_size = (i % 2 == 0) ? 40 : 20;
        }
        else {

            pop_size = (i % 2 == 0) ? Config::EXPLORATION_POPULATION_SIZE
                : Config::EXPLOITATION_POPULATION_SIZE;
        }

        islands_[i] = new Island(evaluators_[i], data, pop_size, i);
    }
}

Optimizer::~Optimizer() {
    StopThreads();

    for (int i = 0; i < num_islands_; ++i) {
        delete islands_[i];
        delete evaluators_[i];
    }
    delete[] island_generations_;
}

void Optimizer::StopThreads() {
    is_running_ = false;

    for (auto& t : worker_threads_) {
        if (t.joinable())
            t.join();
    }
    worker_threads_.clear();
}

void Optimizer::Initialize() {
    StopThreads();

    for (int i = 0; i < num_islands_; ++i) {

        INITIALIZATION_TYPE strategy;
        switch (i) {
        case 0: strategy = INITIALIZATION_TYPE::BIN_PACKING; break;
        case 1: strategy = INITIALIZATION_TYPE::BIN_PACKING; break;
        case 2: strategy = INITIALIZATION_TYPE::BIN_PACKING; break;
        case 3: strategy = INITIALIZATION_TYPE::BIN_PACKING; break;
        case 4: strategy = INITIALIZATION_TYPE::RANDOM; break;
        case 5: strategy = INITIALIZATION_TYPE::RANDOM; break;
        default: strategy = INITIALIZATION_TYPE::BIN_PACKING; break;
        }
        islands_[i]->Initialize(strategy);

        if (islands_[i]->GetBestFitness() < current_best_fitness_) {
            current_best_ = islands_[i]->GetBestSolution();
            current_best_fitness_ = islands_[i]->GetBestFitness();
            current_best_indiv_ = islands_[i]->GetBestIndividual();
        }
    }

    is_running_ = true;
    start_time_ = std::chrono::steady_clock::now();

    for (int i = 0; i < num_islands_; ++i) {
        islands_[i]->SetStartTime(start_time_);
    }

    for (int i = 0; i < num_islands_; ++i) {
        int pred = (i + num_islands_ - 1) % num_islands_;
        islands_[i]->SetRingPredecessor(islands_[pred]);
    }


    for (int i = 0; i < num_islands_; ++i) {
        if (i % 2 == 1) {

            std::vector<Island*> siblings;
            for (int j = 1; j < num_islands_; j += 2) {
                if (j != i) {
                    siblings.push_back(islands_[j]);
                }
            }

            if (i - 1 >= 0) {
                siblings.push_back(islands_[i - 1]);
            }
            islands_[i]->SetExploitSiblings(siblings);
        }
        else {

            if (i + 1 < num_islands_) {
                islands_[i]->SetExploitSiblings({ islands_[i + 1] });
            }
        }
    }

    for (int i = 0; i < num_islands_; ++i) {
        worker_threads_.emplace_back(&Optimizer::IslandWorkerLoop, this, i);
    }

    if (Config::SHOW_LOGS) {
        cout << "[OPT] Started " << num_islands_ << " island workers in RING topology\n";
        cout << "[OPT] Even indices = EXPLORATION | Odd indices = EXPLOITATION\n";
        cout << "[OPT] EXPLOIT islands broadcast best to siblings (non-native)\n";
        cout << "[OPT] Migration: ASYNC pull-based (islands pull when stuck)\n";
    }
}

void Optimizer::IslandWorkerLoop(int island_idx) {
    Island* my_island = islands_[island_idx];

    long long local_gen = 0;
    auto last_log_time = std::chrono::steady_clock::now();
    long long gens_since_log = 0;

    while (is_running_) {
        my_island->RunGeneration();
        local_gen++;
        gens_since_log++;
        island_generations_[island_idx]++;

        if (my_island->GetBestFitness() < current_best_fitness_) {
            std::lock_guard<std::mutex> lock(global_mutex_);
            if (my_island->GetBestFitness() < current_best_fitness_) {


                double diff_pct = 0.0;
                std::vector<double> pcts;
                int num_groups = evaluator_.GetNumGroups();
                Individual new_best_indiv_copy;

                if (Config::SHOW_LOGS) {
                    if (current_best_indiv_.GetGenotype().size() > 0) {
                        const std::vector<int>& g1 = current_best_indiv_.GetGenotype();
                        Individual new_best = my_island->GetBestIndividual();
                        const std::vector<int>& g2 = new_best.GetGenotype();
                        const std::vector<int>& perm = evaluator_.GetProblemData().GetPermutation();
                        int size = (int)g1.size();

                        if (size > 0 && g1.size() == g2.size()) {
                            std::vector<int> p1(size, -2), p2(size, -2);
                            std::vector<int> last1(num_groups, -1), last2(num_groups, -1);

                            for (int customer_id : perm) {
                                int idx = customer_id - 2;
                                if (idx < 0 || idx >= size) continue;

                                int gr1 = g1[idx];
                                if (gr1 >= 0 && gr1 < num_groups) {
                                    p1[idx] = last1[gr1];
                                    last1[gr1] = idx;
                                }

                                int gr2 = g2[idx];
                                if (gr2 >= 0 && gr2 < num_groups) {
                                    p2[idx] = last2[gr2];
                                    last2[gr2] = idx;
                                }
                            }

                            int dist = 0;
                            for (int k = 0; k < size; ++k) {
                                if (p1[k] != p2[k]) dist++;
                            }
                            diff_pct = (dist * 100.0) / size;
                        }
                    }
                    new_best_indiv_copy = my_island->GetBestIndividual();
                }

                current_best_ = my_island->GetBestSolution();
                current_best_fitness_ = my_island->GetBestFitness();
                current_best_indiv_ = my_island->GetBestIndividual();

                if (Config::SHOW_LOGS) {
                    auto now = std::chrono::steady_clock::now();
                    double elapsed = std::chrono::duration<double>(now - start_time_).count();
                    const char* island_type = my_island->IsExploration() ? "EXPLORE" : "EXPLOIT";

                    const ProblemData& pd = evaluator_.GetProblemData();
                    const std::vector<int>& genotype = current_best_indiv_.GetGenotype();
                    const std::vector<int>& demands = pd.GetDemands();
                    int capacity = pd.GetCapacity();

                    std::vector<int> group_load(num_groups, 0);
                    for (size_t ci = 0; ci < genotype.size(); ++ci) {
                        int group = genotype[ci];
                        if (group >= 0 && group < num_groups) {
                            int customer_id = static_cast<int>(ci) + 2;
                            int demand_idx = customer_id - 1;
                            if (demand_idx >= 0 && demand_idx < static_cast<int>(demands.size())) {
                                group_load[group] += demands[demand_idx];
                            }
                        }
                    }

                    pcts.resize(num_groups);
                    for (int g = 0; g < num_groups; ++g) {
                        pcts[g] = (capacity > 0) ? (100.0 * group_load[g] / capacity) : 0.0;
                    }
                    std::sort(pcts.begin(), pcts.end(), std::greater<double>());

                    cout << "\033[92m [Island " << island_idx << " " << island_type
                        << "] NEW BEST: " << fixed << setprecision(2)
                        << current_best_fitness_
                        << " (ret=" << current_best_indiv_.GetReturnCount()
                        << ", diff=" << setprecision(1) << diff_pct << "%)"
                        << " @ gen " << local_gen << " (t=" << setprecision(1) << elapsed
                        << "s)\033[0m [";
                    for (int g = 0; g < num_groups; ++g) {
                        if (g > 0) cout << ", ";
                        if (pcts[g] > 100.0) {
                            cout << "\033[91m" << setprecision(0) << pcts[g] << "%\033[0m";
                        }
                        else if (pcts[g] > 95.0) {
                            cout << "\033[93m" << setprecision(0) << pcts[g] << "%\033[0m";
                        }
                        else {
                            cout << setprecision(0) << pcts[g] << "%";
                        }
                    }
                    cout << "]\n";
                }
                else {
                    cout << fixed << setprecision(2) << current_best_fitness_ << endl;
                }
            }
        }



        auto now = std::chrono::steady_clock::now();
        double since_log =
            std::chrono::duration<double>(now - last_log_time).count();
        if (since_log >= Config::LOG_INTERVAL_SECONDS && Config::SHOW_LOGS) {
            double gen_per_sec = gens_since_log / since_log;
            const char* island_type =
                my_island->IsExploration() ? "EXPLORE" : "EXPLOIT";
            cout << " [I" << island_idx << " " << island_type << "] "
                << gens_since_log << " gens in " << setprecision(1) << since_log
                << "s = " << setprecision(1) << gen_per_sec
                << " gen/s, best=" << setprecision(2) << my_island->GetBestFitness()
                << "\n";

            last_log_time = now;
            gens_since_log = 0;
        }
    }
}




void Optimizer::PrintIslandStats() {
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();

    cout << "\n=== Ring Island Statistics (t=" << fixed << setprecision(1)
        << elapsed << "s) ===\n";
    long long total_gens = 0;
    long long totalpr = 0;
    for (int i = 0; i < num_islands_; ++i) {
        long long gens = island_generations_[i];
        total_gens += gens;
        double rate = gens / std::max(0.1, elapsed);
        const char* type = (i % 2 == 0) ? "EXPLORE" : "EXPLOIT";
        cout << "  I" << i << " (" << type << "): " << gens << " gens ("
            << setprecision(1) << rate << " gen/s)\n";
    }
    cout << "  TOTAL: " << total_gens << " gens (" << setprecision(1)
        << (total_gens / std::max(0.1, elapsed)) << " gen/s)\n";


    long long total_hits = 0, total_misses = 0;
    long long route_hits = 0, route_misses = 0;
    for (int i = 0; i < num_islands_; ++i) {
        totalpr += islands_[i]->getPRStats();
        total_hits += islands_[i]->GetCacheHits();
        total_misses += islands_[i]->GetCacheMisses();


        route_hits += evaluators_[i]->GetRouteCacheHits();
        route_misses += evaluators_[i]->GetRouteCacheMisses();
    }

    double hit_rate = (total_hits + total_misses > 0)
        ? (100.0 * total_hits / (total_hits + total_misses))
        : 0.0;

    double route_hit_rate =
        (route_hits + route_misses > 0)
        ? (100.0 * route_hits / (route_hits + route_misses))
        : 0.0;

    cout << "  Solution Cache (L1): " << total_hits << " hits / " << total_misses
        << " misses (" << setprecision(1) << hit_rate << "% hit rate)\n";
    cout << "  Route Cache (L2):    " << route_hits << " hits / " << route_misses
        << " misses (" << setprecision(1) << route_hit_rate << "% hit rate)\n";

    cout << "  Global Best: " << setprecision(2) << current_best_fitness_ << "\n";
    cout << "  Path Relinking Successes: " << totalpr << "\n";


    std::vector<uint64_t> best_hashes(num_islands_);
    std::vector<double> best_fits(num_islands_);
    for (int i = 0; i < num_islands_; ++i) {
        Individual best = islands_[i]->GetBestIndividual();
        best_fits[i] = best.GetFitness();
        uint64_t h = 1469598103934665603ULL;
        for (int x : best.GetGenotype()) {
            h ^= x;
            h *= 1099511628211ULL;
        }
        best_hashes[i] = h;
    }

    std::set<uint64_t> unique_bests(best_hashes.begin(), best_hashes.end());
    int unique_best_count = (int)unique_bests.size();


    int similar_fit_count = 0;
    for (int i = 0; i < num_islands_; ++i) {
        for (int j = i + 1; j < num_islands_; ++j) {
            if (std::abs(best_fits[i] - best_fits[j]) < 1000)
                similar_fit_count++;
        }
    }

    std::string global_issues = "";
    if (unique_best_count <= 2)
        global_issues +=
        "\033[31m[HOMOGENIZED: " + std::to_string(unique_best_count) +
        "/" + std::to_string(num_islands_) + " unique bests]\033[0m ";

    int total_pairs = num_islands_ * (num_islands_ - 1) / 2;
    if (similar_fit_count >= total_pairs * 2 / 3)
        global_issues +=
        "\033[33m[CONVERGED: " + std::to_string(similar_fit_count) +
        "/" + std::to_string(total_pairs) + " pairs similar]\033[0m ";

    double explore_avg = 0, exploit_avg = 0;
    int explore_count = 0, exploit_count = 0;
    for (int i = 0; i < num_islands_; ++i) {
        if (i % 2 == 0) {
            explore_avg += best_fits[i];
            explore_count++;
        }
        else {
            exploit_avg += best_fits[i];
            exploit_count++;
        }
    }
    if (explore_count > 0) explore_avg /= explore_count;
    if (exploit_count > 0) exploit_avg /= exploit_count;

    if (explore_avg < exploit_avg - 1000)
        global_issues += "\033[35m[EXPLORE_BETTER: exploiters lagging]\033[0m ";

    if (!global_issues.empty()) {
        cout << "  ISSUES: " << global_issues << "\n";
    }

    cout << "===============================\n\n";
}

void Optimizer::RunIteration() {
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();

    static double last_global_log = 0.0;
    if (elapsed - last_global_log >= Config::LOG_INTERVAL_SECONDS) {
        if (Config::SHOW_LOGS) {
            PrintIslandStats();
        }
        last_global_log = elapsed;
    }

}

int Optimizer::GetGeneration() {
    long long total = 0;
    for (int i = 0; i < num_islands_; ++i) {
        total += island_generations_[i];
    }
    return static_cast<int>(total / num_islands_);
}
