/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "config.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <memory>
#include <cmath>
#include <array>
#include <thread>

#include "Utils.h"
#include "FastBoard.h"
#include "Random.h"
#include "Network.h"
#include "GTP.h"
#include "Utils.h"
#include "NNCache.h"
#include "../simplemodel/model.h"


using namespace Utils;


static std::shared_ptr<dlib::zero_model> model;

bool Network::load_weights(const std::string& path) {
    if (!model)
        model = std::make_shared<dlib::zero_model>();
    return model->load_weights(path);
}

std::mutex Network::mtx_;


// Rotation helper
static std::array<std::array<int, 361>, 8> rotate_nn_idx_table;



void Network::initialize(void) {
    // Prepare rotation table
    for(auto s = 0; s < 8; s++) {
        for(auto v = 0; v < 19 * 19; v++) {
            rotate_nn_idx_table[s][v] = rotate_nn_idx(v, s);
        }
    }
}

Network::Netresult Network::get_scored_moves(
    const GameState * state, Ensemble ensemble, int rotation, bool skip_cache) {
    Netresult result;

    NNPlanes planes;
    gather_features(state, planes);

    if (ensemble == DIRECT) {
        assert(rotation >= 0 && rotation <= 7);
        result = get_scored_moves_internal(planes, rotation, skip_cache);
    } else {
        assert(ensemble == RANDOM_ROTATION);
        assert(rotation == -1);
        auto rand_rot = Random::get_Rng().randfix<8>();
        result = get_scored_moves_internal(planes, rand_rot, skip_cache);
    }

    return result;
}


Network::Netresult Network::nn_predict(const NNPlanes & planes, bool skip_cache) {
    // See if we already have this in the cache.
    if (!skip_cache) {
        Netresult result;
        if (NNCache::get_NNCache().lookup(planes, result)) {
          return result;
        }
    }

    std::lock_guard<std::mutex> lock(mtx_);
    auto result = model->predict(planes, cfg_softmax_temp);
    if (!skip_cache) {
        // Insert result into cache.
        NNCache::get_NNCache().insert(planes, result);
    }
    return result;
}

Network::Netresult Network::get_scored_moves_internal(
    const NNPlanes & planes, int rotation, bool skip_cache) {

    if (rotation == 0) {
        auto out = nn_predict(planes, skip_cache);
        // Sigmoid
        float winrate = (1.0f + out.second) / 2.0f;
        out.second = winrate;
        return out;
    }

    assert(rotation >= 0 && rotation <= 7);
    assert(INPUT_CHANNELS == planes.size());
        
    // Data layout is input_data[(c * height + h) * width + w]
    NNPlanes inputs;
    for (int c = 0; c < INPUT_CHANNELS; ++c) {
        for (int idx = 0; idx < FastBoard::BOARDSQ; ++idx) {
            auto rot_idx = rotate_nn_idx_table[rotation][idx];
            inputs[c][idx] = planes[c][rot_idx];
        }
    }
   
    std::vector<float> result(FastBoard::BOARDSQ+1);

    auto out = nn_predict(inputs, skip_cache);

    float winrate_out = out.second;
    auto outputs = out.first;

    for (auto idx = size_t{0}; idx < outputs.size(); idx++) {
        auto val = outputs[idx];
        if (idx < FastBoard::BOARDSQ) {
                
            auto rot_idx = rotate_nn_idx_table[rotation][idx];
            result[rot_idx] = val;
        } else {
            result[FastBoard::BOARDSQ] = val;
        }
    }

    // Sigmoid
    float winrate_sig = (1.0f + winrate_out) / 2.0f;
        
    return std::make_pair(result, winrate_sig);
}


void Network::fill_input_plane_pair(const FastBoard::StoneMap& board,
    BoardPlane& black, BoardPlane& white) {
    for (int idx = 0; idx < FastBoard::BOARDSQ; idx++) {
            auto color = board[idx];
            if (color != FastBoard::EMPTY) {
                if (color == FastBoard::BLACK) {
                    black[idx] = true;
                } else {
                    white[idx] = true;
                }
            }
    }
}

void Network::gather_features(const GameState* state, NNPlanes & planes) {

    BoardPlane& black_to_move = planes[2 * INPUT_MOVES];
    BoardPlane& white_to_move = planes[2 * INPUT_MOVES + 1];

    const auto to_move = state->get_to_move();
    const auto blacks_move = to_move == FastBoard::BLACK;

    const auto black_offset = blacks_move ? 0 : INPUT_MOVES;
    const auto white_offset = blacks_move ? INPUT_MOVES : 0;

    if (blacks_move) {
        black_to_move.set();
    } else {
        white_to_move.set();
    }

    const auto moves = std::min<size_t>(state->get_movenum() + 1, INPUT_MOVES);
    // Go back in time, fill history boards
    for (auto h = size_t{0}; h < moves; h++) {
        // collect white, black occupation planes
        fill_input_plane_pair(state->get_past_board(h),
                              planes[black_offset + h],
                              planes[white_offset + h]);
    }
}

int Network::rotate_nn_idx(const int vertex, int symmetry) {
    assert(vertex >= 0 && vertex < 19*19);
    assert(symmetry >= 0 && symmetry < 8);
    int x = vertex % 19;
    int y = vertex / 19;
    int newx;
    int newy;

    if (symmetry >= 4) {
        std::swap(x, y);
        symmetry -= 4;
    }

    if (symmetry == 0) {
        newx = x;
        newy = y;
    } else if (symmetry == 1) {
        newx = x;
        newy = 19 - y - 1;
    } else if (symmetry == 2) {
        newx = 19 - x - 1;
        newy = y;
    } else {
        assert(symmetry == 3);
        newx = 19 - x - 1;
        newy = 19 - y - 1;
    }

    int newvtx = (newy * 19) + newx;
    assert(newvtx >= 0 && newvtx < 19*19);
    return newvtx;
}



