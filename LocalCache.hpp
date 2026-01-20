#include <vector>
#include <cstdint>
#include <cstddef> 

struct LocalCacheEntry {
    uint64_t signature = 0;
    uint32_t checksum = 0;
    double fitness = -1.0;
    int returns = 0;
    bool occupied = false;
};

class LocalCache {
private:



    static constexpr size_t SIZE = 1 << 25;
    static constexpr size_t MASK = SIZE - 1;

    size_t occupied_count_ = 0;
    std::vector<LocalCacheEntry> table_;

    std::vector<int> history_counts_;
    int genotype_size_ = 0;


    uint64_t CalculateSignature(const std::vector<int>& vec) const {
        uint64_t seed = vec.size();
        for (int val : vec) {

            seed ^= val + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        }
        return seed;
    }



    uint32_t CalculateChecksum(const std::vector<int>& vec) const {
        uint32_t sum = 0;

        for (size_t i = 0; i < vec.size(); ++i) {
            sum += static_cast<uint32_t>(vec[i]) * (static_cast<uint32_t>(i) + 1);
        }
        return sum;
    }

public:
    long long colisions = 0;

    LocalCache() {
        table_.resize(SIZE);
    }

    void InitHistory(int size) {
        if (history_counts_.empty()) {
            genotype_size_ = size;
            history_counts_.resize(size, 0);
        }
    }

    int GetFrequency(int index) const {
        if (index >= 0 && index < (int)history_counts_.size()) {
            return history_counts_[index];
        }
        return 0;
    }

    void UpdateHistory(const std::vector<int>& genotype) {
        if (genotype.size() != history_counts_.size()) return;

        for (size_t i = 0; i < genotype.size() - 1; ++i) {
            if (genotype[i] == genotype[i + 1]) {
                history_counts_[i]++;
            }
        }
    }

    size_t GetSize() const {
        return occupied_count_;
    }

    void Clear() {
        for (auto& entry : table_) {
            entry.occupied = false;
        }
        occupied_count_ = 0;
        colisions = 0;
    }

    void ClearHistory() {
        std::fill(history_counts_.begin(), history_counts_.end(), 0);
    }


    bool TryGet(const std::vector<int>& genotype, double& out_fitness, int& out_returns) {
        uint64_t sig = CalculateSignature(genotype);
        size_t idx = sig & MASK;

        const auto& entry = table_[idx];

        if (!entry.occupied) return false;


        if (entry.signature == sig) {
            uint32_t check = CalculateChecksum(genotype);
            if (entry.checksum == check) {
                out_fitness = entry.fitness;
                out_returns = entry.returns;
                return true;
            }
        }

        return false;
    }

    void Insert(const std::vector<int>& genotype, double fitness, int returns) {
        uint64_t sig = CalculateSignature(genotype);
        uint32_t check = CalculateChecksum(genotype);
        size_t idx = sig & MASK;

        LocalCacheEntry& entry = table_[idx];

        if (!entry.occupied) {
            occupied_count_++;
        }
        else {

            if (entry.signature == sig && entry.checksum == check) {


            }
            else {

                colisions++;
            }
        }


        entry.signature = sig;
        entry.checksum = check;
        entry.fitness = fitness;
        entry.returns = returns;
        entry.occupied = true;
    }
};