#include "Evaluator.hpp"
#include "Optimizer.hpp"
#include "ProblemLoader.hpp"
#include <iostream>

using namespace LcVRPContest;

void StartOptimization(const string& folder_name, const string& instance_name, int max_iterations, bool use_random_permutation = true) {
	ProblemLoader problem_loader(folder_name, instance_name, use_random_permutation);
	ProblemData problem_data = problem_loader.LoadProblem();

	int num_groups = problem_data.GetNumGroups();
	Evaluator evaluator(problem_data, num_groups);
	Optimizer optimizer(evaluator);

	optimizer.Initialize();

	for (int i = 0; i < max_iterations; ++i) {
		optimizer.RunIteration();
	}

	vector<int>* best_solution = optimizer.GetCurrentBest();
	double best_fitness = evaluator.Evaluate(*best_solution);
	cout << "final best fitness: " << best_fitness << endl;
}
void StartOptimizationTimeBased(const string& folder_name, const string& instance_name, int max_iterations, bool use_random_permutation = true) {
	ProblemLoader problem_loader(folder_name, instance_name, use_random_permutation);
	ProblemData problem_data = problem_loader.LoadProblem();

	int num_groups = problem_data.GetNumGroups();
	Evaluator evaluator(problem_data, num_groups);
	Optimizer optimizer(evaluator);

	optimizer.Initialize();

	int max_seconds = 60 * 60 * 2; // 2hours

	auto start_time = std::chrono::steady_clock::now();
	auto end_time = start_time + std::chrono::seconds(max_seconds);

	while (std::chrono::steady_clock::now() < end_time) {
		optimizer.RunIteration();
	}


	vector<int>* best_solution = optimizer.GetCurrentBest();
	double best_fitness = evaluator.Evaluate(*best_solution);
	cout << "final best fitness: " << best_fitness << endl;
}


int main() {
	int max_iterations = 10;
	bool use_random_permutation = false;

	StartOptimizationTimeBased("Vrp-Set-D", "ORTEC-n323-k21", max_iterations, use_random_permutation);

	return 0;
}