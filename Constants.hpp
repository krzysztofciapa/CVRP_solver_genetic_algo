#pragma once

namespace LcVRPContest {
    namespace Config {


        constexpr long long MAX_TIME_SECONDS = 80 * 60;
        constexpr double LOG_INTERVAL_SECONDS = 30.0;
        constexpr int STAGNATION_THRESHOLD = 300;
        constexpr bool SHOW_LOGS = false;


        constexpr int LARGE_INSTANCE_THRESHOLD = 1500;
        constexpr int HUGE_INSTANCE_THRESHOLD = 3000;

        constexpr double EXPLOITATION_VND_PROB_LARGE = 0.95;
        constexpr double EXPLORATION_VND_PROB_LARGE = 0.20;






        constexpr double MIGRATION_INTERVAL_HEALTHY = 15.0;
        constexpr double MIGRATION_INTERVAL_SICK = 3.0;


        constexpr double HEALTH_CRITICAL = 0.15;
        constexpr double HEALTH_SICK = 0.30;



        constexpr double TRICKLE_PROB_HEALTHY = 0.03;
        constexpr double TRICKLE_PROB_SICK = 0.15;


        constexpr double BROADCAST_WARMUP_SECONDS = 15.0;


        constexpr int EXPLORATION_POPULATION_SIZE = 70;
        constexpr int EXPLORATION_TOURNAMENT_SIZE = 2;
        constexpr double EXPLORATION_MUTATION_PROB = 0.50;
        constexpr int EXPLORATION_VND_MIN = 5;
        constexpr int EXPLORATION_VND_MAX = 15;
        constexpr double EXPLORATION_VND_PROB =
            0.60;
        constexpr double EXPLORATION_VND_EXTRA_PROB =
            0.35;



        constexpr double EXPLORE_I0_MUT_MIN = 0.20;
        constexpr double EXPLORE_I0_MUT_MAX = 0.60;



        constexpr double EXPLORE_I2_MUT_MIN = 0.30;
        constexpr double EXPLORE_I2_MUT_MAX = 0.65;


        constexpr double EXPLORE_I4_MUT_MIN = 0.40;
        constexpr double EXPLORE_I4_MUT_MAX = 0.70;


        constexpr int EXPLOITATION_POPULATION_SIZE = 20;
        constexpr int EXPLOITATION_TOURNAMENT_SIZE = 4;
        constexpr double EXPLOITATION_MUTATION_PROB = 0.05;
        constexpr int EXPLOITATION_VND_MIN = 30;
        constexpr int EXPLOITATION_VND_MAX = 60;
        constexpr double EXPLOITATION_VND_PROB = 0.85;



        constexpr double EXPLOIT_I1_OP_PR_PROB = 0.60;
        constexpr double EXPLOIT_I1_OP_RR_PROB = 0.30;
        constexpr double EXPLOIT_I1_OP_SPLIT_PROB = 0.10;




        constexpr double EXPLOIT_I3_OP_PR_PROB = 0.50;
        constexpr double EXPLOIT_I3_OP_RR_PROB = 0.40;
        constexpr double EXPLOIT_I3_OP_SPLIT_PROB = 0.10;




        constexpr double EXPLOIT_I5_OP_PR_PROB = 0.60;
        constexpr double EXPLOIT_I5_OP_RR_PROB = 0.30;
        constexpr double EXPLOIT_I5_OP_SPLIT_PROB = 0.10;




        constexpr double EXPLOITATION_P_SWAP3 = 0.30;
        constexpr double EXPLOITATION_P_SWAP4 = 0.20;



        constexpr double ADAPT_EPSILON =
            0.25;


        constexpr double ENDGAME_THRESHOLD = 0.85;





        constexpr double ADAPTIVE_MUT_MIN = 0.05;
        constexpr double ADAPTIVE_MUT_MAX = 0.80;
        constexpr double ADAPTIVE_CHAOS_BOOST = 0.30;
        constexpr double ADAPTIVE_CHAOS_PENALTY = 0.10;




        constexpr double CATASTROPHE_STAGNATION_SECONDS = 300.0;
        constexpr double EXPLOIT_CATASTROPHE_STAGNATION_SECONDS = 180.0;

        constexpr double CATASTROPHE_HEALTH_THRESHOLD = 0.10;

        constexpr double VND_EXHAUSTED_THRESHOLD = 3.0;
        constexpr double EXPLOIT_VND_EXHAUSTED_THRESHOLD =
            20.0;
        constexpr int VND_EXHAUSTED_MIN_CALLS = 200;

        constexpr int CATASTROPHE_VND_ITERS = 30;



        constexpr double EXPLOIT_RR_STAGNATION_SECONDS = 20.0;
        constexpr double EXPLOIT_RR_INTERVAL_SECONDS = 5.0;
        constexpr double EXPLOIT_RR_HEALTH_THRESHOLD = 0.25;
        constexpr double EXPLOIT_HEAVY_RR_INTENSITY = 0.45;
        constexpr double EXPLOIT_RR_INTENSITY_EXTREME = 0.60;
        constexpr double EXPLOIT_EXTREME_STAGNATION_SECONDS = 90.0;

        constexpr long long EXPLOIT_RR_STAGNATION_TRIGGER = 200;
        constexpr long long EXPLOIT_RR_INTERVAL = 50;
        constexpr long long CATASTROPHE_MIN_GAP_GENS = 500;


        constexpr bool ENABLE_FRANKENSTEIN = true;
        constexpr int FRANKENSTEIN_BEAM_WIDTH = 50;
        constexpr int FRANKENSTEIN_VND_ITERS = 40;
        constexpr int FRANKENSTEIN_VND_ITERS_LATE = 60;
        constexpr int FRANKENSTEIN_VND_PASSES = 3;
        constexpr double FRANKENSTEIN_FORCE_INJECT_PROB = 0.10;
        constexpr int FRANKENSTEIN_MAX_SOURCE_ROUTES = 5000;
        constexpr int FRANKENSTEIN_MAX_INSTANCE_SIZE = 6000;




        constexpr double ELITE_RATIO_EXPLORATION_LOW = 0.05;
        constexpr double ELITE_RATIO_EXPLORATION_HIGH = 0.20;
        constexpr double ELITE_RATIO_EXPLOITATION_LOW = 0.30;
        constexpr double ELITE_RATIO_EXPLOITATION_HIGH = 0.90;

        constexpr int ELITERATIO = 1;


        constexpr double RUIN_BASE_PCT = 0.10;
        constexpr double RUIN_INTENSITY_SCALE = 0.40;
        constexpr int RUIN_MIN_REMOVED = 5;
        constexpr double RUIN_BASE_PCT_EXPLOITATION = 0.10;
        constexpr double RUIN_INTENSITY_SCALE_EXPLOITATION =
            0.15;
        constexpr double EXPLOITATION_MIN_MICROSPLIT = 0.20;


        constexpr double EJECTION_PROBABILITY = 0.20;
        constexpr double PATH_RELINK_PROBABILITY = 0.15;


        constexpr int MIGRATION_ELITE_COUNT = 2;
        constexpr int MIGRATION_DIVERSE_COUNT =
            3;


        constexpr double PROGRESS_IMMUNITY_SECONDS =
            10.0;


        constexpr bool ALLOW_SWAP = true;
        constexpr bool ALLOW_3SWAP = true;
        constexpr bool ALLOW_EJECTION = true;
        constexpr bool ALLOW_LOAD_BALANCING = false;


        constexpr size_t ROUTE_POOL_MAX_SIZE = 10000;
        constexpr double GREEDY_ASSEMBLER_INTERVAL_SECONDS = 8.0;
        constexpr int GREEDY_NUM_STARTS = 10;
        constexpr size_t MIN_ROUTE_SIZE_FOR_POOL = 2;




        constexpr int NUM_NEIGHBORS = 20;
        constexpr int HISTORY_LAMBDA = 100;
        constexpr bool split = true;



        constexpr double SPLIT_ROUTE_PENALTY = 1e10;
    }
}
