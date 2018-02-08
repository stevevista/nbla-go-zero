#pragma once
#include "defs.hpp"
#include "convert.hpp"
#include <unordered_map>
#include <list>
#include <nblapp/solver.hpp>
#include <nblapp/variable.hpp>


using namespace nblapp;
using std::shared_ptr;
using std::vector;


class MODEL_API ZeroTrainModel {

public:
    typedef vector<float> input_t;

    static constexpr int boardsize = 19;
    static constexpr int residual_layers = 19;
    static constexpr int input_planes = 18;

    ZeroTrainModel(int batch_size);

    float train_batch(vector<MoveData>::const_iterator begin, vector<MoveData>::const_iterator end, 
        BaseSolver& solver, float weight_decay = 0, bool random_rotate=true);

    bool load_weights(const std::string& path);
    void save_weights(const std::string& path, int seen = 0, int batchs = 0, int rounds = 0, float lr = 0);
    
    int get_batch_size() const { return batch_size_; }
    float get_learning_rate() const { return learning_rate_; }
    int get_seen() const { return seen_; }
    int get_batchs() const { return batchs_; }
    int get_rounds() const { return rounds_; }

protected:
    void get_training_model();

    Variable input;
    Variable label;
    Variable result;
    Variable loss;

    int batch_size_;
    float learning_rate_;
    int seen_;
    int batchs_;
    int rounds_;

};
