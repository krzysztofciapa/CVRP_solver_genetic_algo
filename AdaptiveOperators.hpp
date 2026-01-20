#pragma once
#include <random>
#include <array>
#include <algorithm>
#include <cmath>

namespace LcVRPContest {


    enum class OperatorGroup {
        VND,
        MUTATION,
        MICROSPLIT,
        LNS,
        HEAVY,
        COUNT
    };


    enum class LNSStrategy {
        RANDOM_CLUSTER,
        WORST_ROUTES,
        RELATED_REMOVAL,
        PERMUTATION_BASED,
        TARGET_OVERFLOW,
        COUNT
    };

    class AdaptiveOperatorSelector {
    public:
        static constexpr int NUM_GROUPS = static_cast<int>(OperatorGroup::COUNT);
        static constexpr int NUM_LNS = static_cast<int>(LNSStrategy::COUNT);
        static constexpr double MIN_PROBABILITY = 0.05;
        static constexpr double DECAY_FACTOR = 0.95;
        static constexpr int UPDATE_INTERVAL = 100;

        AdaptiveOperatorSelector() {
            Reset();
        }

        void Reset() {
            for (int i = 0; i < NUM_GROUPS; ++i) {
                group_usage_[i] = 0;
                group_success_[i] = 0;
                group_improvement_[i] = 0.0;
                group_score_[i] = 1.0;
            }
            for (int i = 0; i < NUM_LNS; ++i) {
                lns_usage_[i] = 0;
                lns_success_[i] = 0;
                lns_improvement_[i] = 0.0;
                lns_score_[i] = 1.0;
            }
            generation_counter_ = 0;
        }


        void RecordResult(OperatorGroup group, bool success, double improvement = 0.0) {
            int g = static_cast<int>(group);
            group_usage_[g]++;
            if (success) {
                group_success_[g]++;
                group_improvement_[g] += improvement;
            }
        }

        void RecordLNSResult(LNSStrategy strategy, bool success, double improvement = 0.0) {
            int s = static_cast<int>(strategy);
            lns_usage_[s]++;
            if (success) {
                lns_success_[s]++;
                lns_improvement_[s] += improvement;
            }
        }


        double GetGroupProbability(OperatorGroup group) const {
            double total_score = 0.0;
            for (int i = 0; i < NUM_GROUPS; ++i) {
                total_score += group_score_[i];
            }
            if (total_score <= 0.0) return 1.0 / NUM_GROUPS;

            double raw_prob = group_score_[static_cast<int>(group)] / total_score;

            return std::max(MIN_PROBABILITY, raw_prob);
        }


        LNSStrategy SelectLNSStrategy(std::mt19937& rng) const {
            double total_score = 0.0;
            for (int i = 0; i < NUM_LNS; ++i) {
                total_score += lns_score_[i];
            }

            std::uniform_real_distribution<double> dist(0.0, total_score);
            double r = dist(rng);

            double cumulative = 0.0;
            for (int i = 0; i < NUM_LNS; ++i) {
                cumulative += lns_score_[i];
                if (r <= cumulative) {
                    return static_cast<LNSStrategy>(i);
                }
            }
            return LNSStrategy::RANDOM_CLUSTER;
        }


        void OnGeneration() {
            generation_counter_++;
            if (generation_counter_ >= UPDATE_INTERVAL) {
                UpdateScores();
                generation_counter_ = 0;
            }
        }


        std::string GetDiagnostics() const {
            std::string result = "AOS[";
            const char* names[] = { "VND", "MUT", "uSPL", "LNS", "HVY" };
            for (int i = 0; i < NUM_GROUPS; ++i) {
                if (i > 0) result += " ";
                double rate = group_usage_[i] > 0 ?
                    (100.0 * group_success_[i] / group_usage_[i]) : 0.0;
                result += names[i];
                result += ":";
                char buf[16];
                snprintf(buf, sizeof(buf), "%.0f%%", rate);
                result += buf;
            }
            result += "]";
            return result;
        }

    private:
        void UpdateScores() {

            for (int i = 0; i < NUM_GROUPS; ++i) {
                if (group_usage_[i] > 0) {
                    double success_rate = static_cast<double>(group_success_[i]) / group_usage_[i];
                    double avg_improvement = group_improvement_[i] / std::max(1, group_success_[i]);



                    double norm_improvement = std::min(1.0, avg_improvement * 100.0);
                    double new_score = success_rate * (1.0 + norm_improvement);


                    group_score_[i] = DECAY_FACTOR * group_score_[i] + (1.0 - DECAY_FACTOR) * new_score;
                    group_score_[i] = std::max(0.1, group_score_[i]);
                }

                group_usage_[i] = 0;
                group_success_[i] = 0;
                group_improvement_[i] = 0.0;
            }


            for (int i = 0; i < NUM_LNS; ++i) {
                if (lns_usage_[i] > 0) {
                    double success_rate = static_cast<double>(lns_success_[i]) / lns_usage_[i];
                    double avg_improvement = lns_improvement_[i] / std::max(1, lns_success_[i]);
                    double norm_improvement = std::min(1.0, avg_improvement * 100.0);
                    double new_score = success_rate * (1.0 + norm_improvement);

                    lns_score_[i] = DECAY_FACTOR * lns_score_[i] + (1.0 - DECAY_FACTOR) * new_score;
                    lns_score_[i] = std::max(0.1, lns_score_[i]);
                }
                lns_usage_[i] = 0;
                lns_success_[i] = 0;
                lns_improvement_[i] = 0.0;
            }
        }


        std::array<int, NUM_GROUPS> group_usage_;
        std::array<int, NUM_GROUPS> group_success_;
        std::array<double, NUM_GROUPS> group_improvement_;
        std::array<double, NUM_GROUPS> group_score_;


        std::array<int, NUM_LNS> lns_usage_;
        std::array<int, NUM_LNS> lns_success_;
        std::array<double, NUM_LNS> lns_improvement_;
        std::array<double, NUM_LNS> lns_score_;

        int generation_counter_ = 0;
    };

}
