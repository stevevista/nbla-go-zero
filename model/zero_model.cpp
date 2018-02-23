#include "zero_model.hpp"
#include <nblapp/parameter.hpp>
#include <nblapp/ops.hpp>
#include <cstring>
#include <cassert>
#include <fstream>
#include "utils.hpp"

using namespace nblapp;

using std::make_shared;


Variable bn_conv2d(const string& name, const Variable& x, 
    const int filters, const std::vector<int>& kernel) {
        
    auto h = nn::conv2d(name, x, filters, kernel, {1, 1}, PAD_SAME, false);
    return nn::batchnorm(name, h, true);
}

void ZeroTrainModel::get_training_model() {
    
    if (!input) {
        
        input = Variable({batch_size_, zero::input_channels, zero::board_size, zero::board_size});
        label = Variable({batch_size_, one_hot_label_ ? 1 : zero::board_moves});
        result = Variable({batch_size_, 1});

        const std::vector<int> kernel = {3, 3};
        const int filters = zero::RESIDUAL_FILTERS;
    
        const int n_classes = zero::board_moves;
    
        auto h = nn::relu(bn_conv2d("init", input, filters, kernel));
    
        for (int i=0; i< zero::RESIDUAL_BLOCKS; i++) {
            ParameterScope _(napp::format_str("block_%d", i+1));
            auto orig = nn::identity(h);
            h = nn::relu(bn_conv2d("0", h, filters, kernel));
            h = nn::relu(nn::add(bn_conv2d("1", h, filters, kernel), orig));
        }
    
        Variable dist, res;
        {
            ParameterScope _("policy_head");
            auto p = nn::relu(bn_conv2d("bn", h, 2, {1, 1}));
            dist = nn::fully_connected("fc", p, n_classes);
        }
    
        {
            ParameterScope _("value_head");
            res = nn::relu(bn_conv2d("bn", h, 1, {1, 1}));
            res = nn::relu(nn::fully_connected("fc1", res, 256));
            res = nn::fully_connected("fc2", res, 1);
            res = nn::tanh(res);
        }

        auto xent = nn::softmax_cross_entropy(dist, label);
        auto res_loss = nn::squared_error(res, result);
        res_loss = nn::reduce_mean(res_loss);

        if (val_scalar_ != 1)
            res_loss = res_loss * val_scalar_;
        
        loss = nn::reduce_mean(xent) + res_loss;
    }
}


ZeroTrainModel::ZeroTrainModel(int batch_size, float val_scalar, bool one_hot_label)
: batch_size_(batch_size)
, val_scalar_(val_scalar)
, one_hot_label_(one_hot_label)
{
    seen_ = 0;
    if (batch_size_ == 0) 
        batch_size_ = 64;
}



float ZeroTrainModel::train_batch(vector<MoveData>::const_iterator begin, vector<MoveData>::const_iterator end, 
    BaseSolver& solver, float weight_decay, bool random_rotate) {

    const int batch_size = std::distance(begin, end);

    const int nclass = label.dim(-1);
    const int boardsize = input.dim(-1);
    const int maxpos = boardsize * boardsize;

    //std::cout << nclass << std::endl;
    //std::cout << boardsize << std::endl;
    //std::cout << batch_size << std::endl;

    auto input_d = input.data_write<float>();
    auto label_d = label.data_write<float>();
    auto result_d = result.data_write<float>();

    for (auto b = 0; b<batch_size; b++) {

        const auto& planes = (begin+b)->input;
        const auto& dist = (begin+b)->probs;
        const auto result = (begin+b)->result;

        {
            int rotate = random_rotate ? (int)rand()%8 : 0;
            for (int idx=0; idx < maxpos; idx++) {
                int rot_idx = rotate_nn_idx_table[rotate][idx];
                for (int c=0; c< zero::input_channels; c++) {
                    input_d[c*maxpos + idx] = (float)planes[c][rot_idx];
                }

                if (!one_hot_label_)
                    label_d[idx] = dist[rot_idx];
                else {
                    if (dist[rot_idx] == 1)
                        label_d[b] = idx;
                }
            }

            if (!one_hot_label_)
                label_d[maxpos] = dist[maxpos];
            else {
                if (dist[maxpos] == 1)
                    label_d[b] = maxpos;
            }
        }

        result_d[b] = float(result);
        input_d += maxpos*18;

        if (!one_hot_label_)
            label_d += nclass;
    }

    loss.train_batch(solver, weight_decay);

    const auto this_loss = loss.data<float>()[0];

    return this_loss;
}


bool ZeroTrainModel::load_weights(const std::string& path) {

    std::ifstream wtfile(path, std::ifstream::binary);
    if (wtfile.fail()) {
        get_training_model();
        return false;
    }

    ParameterScope::load_parameters(wtfile);

    auto status = ParameterScope::get_or_create_constant("__STATUS__", {3}, 0, false);
    auto status_d = status.data<int>();

    seen_ = status_d[0];
    batchs_ = status_d[1];
    rounds_ = status_d[2];

    auto p_lr = ParameterScope::get_or_create_constant("__LEARNING_RATE__", {}, 0.1, false);
    learning_rate_ = p_lr.data<float>()[0];

    auto p_batch = ParameterScope::get_or_create_constant("__BATCH_SIZE__", {}, 32, false);
    int batch_size = p_batch.data<int>()[0];
    if (batch_size_ == 0) {
        batch_size_ = batch_size;
    }

    get_training_model();
    return true;
}


void ZeroTrainModel::save_weights(const std::string& path, int seen, int batchs, int rounds, float lr) {

    if (seen)
        seen_ = seen;

    if (seen)
        batchs_ = seen;

    if (rounds)
        rounds_ = rounds;

    if (lr) 
        learning_rate_ = lr;

    auto status = ParameterScope::get_or_create_constant("__STATUS__", {3}, 0, false);
    auto status_d = status.data_write<int>();
   
    status_d[0] = seen_;
    status_d[1] = batchs_;
    status_d[2] = rounds_;

    auto p_lr = ParameterScope::get_or_create_constant("__LEARNING_RATE__", {}, 0.1, false);
    p_lr.data_write<float>()[0] = learning_rate_;

    auto p_batch = ParameterScope::get_or_create_constant("__BATCH_SIZE__", {}, 32, false);
    p_batch.data_write<int>()[0] = batch_size_;

    std::ofstream ofs(path, std::ofstream::binary);
    ParameterScope::save_parameters(ofs);
    ofs.close();
}
