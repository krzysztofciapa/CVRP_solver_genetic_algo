#pragma once
#include "Island.hpp"
#include "Evaluator.hpp"
#include "LocalSearch.hpp"
#include "ProblemGeometry.hpp"
#include "RoutePool.hpp"
#include "Split.hpp"
#include "ThreadSafeEvaluator.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace LcVRPContest {
	class Optimizer {
	public:
		Optimizer(Evaluator& evaluator);
		~Optimizer();

		void Initialize();
		void RunIteration();
		int GetGeneration();
		Individual GetBestIndividual() { return current_best_indiv_; }
		std::vector<int>* GetCurrentBest() { return &current_best_; }
		double GetCurrentBestFitness() const { return current_best_fitness_; }
		void PrintIslandStats();

	private:
		void IslandWorkerLoop(int island_idx);

		void StopThreads();

		Evaluator& evaluator_;



		std::vector<ThreadSafeEvaluator*> evaluators_;
		std::vector<Island*> islands_;
		int num_islands_;


		std::vector<std::thread> worker_threads_;
		std::atomic<bool> is_running_;


		std::atomic<long long>* island_generations_;
		std::chrono::steady_clock::time_point start_time_;

		std::vector<int> current_best_;
		double current_best_fitness_;
		Individual current_best_indiv_;
		std::mt19937 rng_;
		std::mutex global_mutex_;
	};
}

