// bpred.hpp - direction predictors: static, bimodal, gshare, tournament, perceptron.
// Targets are assumed to come from a perfect BTB / return-address stack (documented limitation).
#pragma once
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace asim {

class BranchPredictor {
public:
    virtual ~BranchPredictor() = default;
    virtual bool predict(uint32_t pc, uint32_t target) = 0;
    virtual void update(uint32_t pc, bool taken) = 0;     // called once per branch with the real outcome
};

class StaticNT : public BranchPredictor {
public:
    bool predict(uint32_t, uint32_t) override { return false; }
    void update(uint32_t, bool) override {}
};

class StaticBTFN : public BranchPredictor {   // backward taken, forward not taken
public:
    bool predict(uint32_t pc, uint32_t target) override { return target < pc; }
    void update(uint32_t, bool) override {}
};

class Bimodal : public BranchPredictor {
public:
    explicit Bimodal(unsigned bits) : mask_((1u << bits) - 1), t_(1u << bits, 1) {}
    bool predict(uint32_t pc, uint32_t) override { return t_[(pc >> 2) & mask_] >= 2; }
    void update(uint32_t pc, bool tk) override {
        uint8_t &c = t_[(pc >> 2) & mask_];
        if (tk) { if (c < 3) ++c; } else if (c > 0) --c;
    }
private:
    uint32_t mask_; std::vector<uint8_t> t_;
};

class Gshare : public BranchPredictor {
public:
    explicit Gshare(unsigned bits) : bits_(bits), mask_((1u << bits) - 1), t_(1u << bits, 1) {}
    bool predict(uint32_t pc, uint32_t) override { return t_[idx(pc)] >= 2; }
    void update(uint32_t pc, bool tk) override {
        uint8_t &c = t_[idx(pc)];
        if (tk) { if (c < 3) ++c; } else if (c > 0) --c;
        ghr_ = ((ghr_ << 1) | (tk ? 1u : 0u)) & mask_;   // history repaired with the true outcome
    }
private:
    uint32_t idx(uint32_t pc) const { return ((pc >> 2) ^ ghr_) & mask_; }
    unsigned bits_; uint32_t mask_, ghr_ = 0; std::vector<uint8_t> t_;
};

// Bimodal vs gshare with a 2-bit per-PC chooser (Alpha 21264 style, simplified).
class Tournament : public BranchPredictor {
public:
    explicit Tournament(unsigned bits) : bim_(bits), gsh_(bits), mask_((1u << bits) - 1), ch_(1u << bits, 1) {}
    bool predict(uint32_t pc, uint32_t tg) override {
        pb_ = bim_.predict(pc, tg); pg_ = gsh_.predict(pc, tg);
        return ch_[(pc >> 2) & mask_] >= 2 ? pg_ : pb_;
    }
    void update(uint32_t pc, bool tk) override {
        uint8_t &c = ch_[(pc >> 2) & mask_];
        if (pb_ != pg_) { if (pg_ == tk) { if (c < 3) ++c; } else if (c > 0) --c; }
        bim_.update(pc, tk); gsh_.update(pc, tk);
    }
private:
    Bimodal bim_; Gshare gsh_; uint32_t mask_; std::vector<uint8_t> ch_; bool pb_ = false, pg_ = false;
};

// Jimenez & Lin perceptron predictor over a 24-bit global history.
class Perceptron : public BranchPredictor {
public:
    explicit Perceptron(unsigned bits) : n_(1u << bits), w_(static_cast<size_t>(n_) * (H + 1), 0) {}
    bool predict(uint32_t pc, uint32_t) override { y_ = dot(pc); return y_ >= 0; }
    void update(uint32_t pc, bool tk) override {
        const int t = tk ? 1 : -1;
        if ((y_ >= 0) != tk || std::abs(y_) <= THETA) {
            int8_t *w = &w_[static_cast<size_t>((pc >> 2) % n_) * (H + 1)];
            w[0] = sat(w[0] + t);
            for (int i = 0; i < H; ++i) w[i + 1] = sat(w[i + 1] + t * (((hist_ >> i) & 1) ? 1 : -1));
        }
        hist_ = ((hist_ << 1) | (tk ? 1u : 0u)) & ((1ull << H) - 1);
    }
private:
    static constexpr int H = 24, THETA = static_cast<int>(1.93 * H + 14);
    static int8_t sat(int v) { return static_cast<int8_t>(v > 127 ? 127 : (v < -128 ? -128 : v)); }
    int dot(uint32_t pc) const {
        const int8_t *w = &w_[static_cast<size_t>((pc >> 2) % n_) * (H + 1)];
        int y = w[0];
        for (int i = 0; i < H; ++i) y += w[i + 1] * (((hist_ >> i) & 1) ? 1 : -1);
        return y;
    }
    uint32_t n_; std::vector<int8_t> w_; uint64_t hist_ = 0; int y_ = 0;
};

inline std::unique_ptr<BranchPredictor> make_predictor(const std::string &name, unsigned bits) {
    if (name == "nt") return std::make_unique<StaticNT>();
    if (name == "btfn") return std::make_unique<StaticBTFN>();
    if (name == "bimodal") return std::make_unique<Bimodal>(bits);
    if (name == "gshare") return std::make_unique<Gshare>(bits);
    if (name == "tournament") return std::make_unique<Tournament>(bits);
    if (name == "perceptron") return std::make_unique<Perceptron>(bits > 4 ? bits - 4 : 1);
    throw std::runtime_error("unknown predictor: " + name);
}

}  // namespace asim
