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
#include "Network.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#endif

#include "UCTNode.h"

#include "FastBoard.h"
#include "FastState.h"
#include "GameState.h"
#include "GTP.h"
#include "NNCache.h"
#include "Random.h"
#include "ThreadPool.h"
#include "Timing.h"
#include "Utils.h"
#include "nn.h"

static std::mutex mtx_;
std::shared_ptr<zero_model> zero_net;

using namespace Utils;

// Rotation helper
static std::array<std::array<int, 361>, 8> rotate_nn_idx_table;

void Network::benchmark(const GameState * state, int iterations) {
    int cpus = cfg_num_threads;
    int iters_per_thread = (iterations + (cpus - 1)) / cpus;

    Time start;

    ThreadGroup tg(thread_pool);
    for (int i = 0; i < cpus; i++) {
        tg.add_task([iters_per_thread, state]() {
            for (int loop = 0; loop < iters_per_thread; loop++) {
                auto vec = get_scored_moves(state, Ensemble::RANDOM_ROTATION, -1, true);
            }
        });
    };
    tg.wait_all();

    Time end;
    auto elapsed = Time::timediff_seconds(start,end);
    myprintf("%5d evaluations in %5.2f seconds -> %d n/s\n",
             iterations, elapsed, (int)(iterations / elapsed));
}

std::pair<int, int> Network::load_network_file(std::string filename) {
    
    if (!zero_net)
        zero_net = std::make_shared<zero_model>();
    if  (!zero_net->load_weights(filename))
        return {0, 0};

    NNCache::get_NNCache().clear();

    return {zero::RESIDUAL_FILTERS, zero::RESIDUAL_BLOCKS};
}

void Network::initialize(void) {
    // Prepare rotation table
    for(auto s = 0; s < 8; s++) {
        for(auto v = 0; v < 19 * 19; v++) {
            rotate_nn_idx_table[s][v] = rotate_nn_idx(v, s);
        }
    }

    // Load network from file
    size_t channels, residual_blocks;
    std::tie(channels, residual_blocks) = load_network_file(cfg_weightsfile);
    if (channels == 0) {
        throw std::runtime_error("weights fail :" + cfg_weightsfile);
    }

    myprintf("Initializing NN backend.\n");
}

Network::Netresult Network::get_scored_moves(
    const GameState* state, Ensemble ensemble, int rotation, bool skip_cache) {
    Netresult result;

    // See if we already have this in the cache.
    if (!skip_cache) {
      if (NNCache::get_NNCache().lookup(state->board.get_hash(), result)) {
        return result;
      }
    }

    if (ensemble == DIRECT) {
        assert(rotation >= 0 && rotation <= 7);
        result = get_scored_moves_internal(state, rotation);
    } else {
        assert(ensemble == RANDOM_ROTATION);
        assert(rotation == -1);
        auto rand_rot = Random::get_Rng().randfix<8>();
        result = get_scored_moves_internal(state, rand_rot);
    }

    // Insert result into cache.
    NNCache::get_NNCache().insert(state->board.get_hash(), result);

    return result;
}

Network::Netresult Network::get_scored_moves_internal(
    const GameState* state, int rotation) {
    assert(rotation >= 0 && rotation <= 7);

    std::pair<std::vector<float>, float>  out;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        gather_features(state, zero_net->input_buffer(), rotation);
        out = zero_net->forward_net(cfg_softmax_temp);
    }
    float winrate_out = out.second;
    auto& outputs = out.first;

    // Sigmoid
    auto winrate_sig = (1.0f + winrate_out) / 2.0f;

    std::vector<scored_node> result;
    for (auto idx = size_t{0}; idx < outputs.size(); idx++) {
        if (idx < FastBoard::BOARDSQ) {
            auto val = outputs[idx];
            auto rot_idx = rotate_nn_idx_table[rotation][idx];
            if (state->board.get_square(rot_idx) == FastBoard::EMPTY) {
                result.emplace_back(val, rot_idx);
            }
        } else {
            result.emplace_back(outputs[idx], FastBoard::PASS);
        }
    }

    return std::make_pair(result, winrate_sig);
}

void Network::gather_features(const GameState* state, float* dest, int rotation) {

    std::fill(dest, dest + INPUT_CHANNELS*FastBoard::BOARDSQ, 0);

    auto black_to_move = dest + (2 * INPUT_MOVES) * FastBoard::BOARDSQ;
    auto white_to_move = black_to_move + FastBoard::BOARDSQ;

    const auto to_move = state->get_to_move();

    const auto moves = std::min<size_t>(state->get_movenum() + 1, INPUT_MOVES);

    // Go back in time, fill history boards
    for (auto h = size_t{0}; h < moves; h++) {

        auto& board = state->get_past_board(h);
        auto me = dest + h * FastBoard::BOARDSQ;
        auto opp = me + INPUT_MOVES * FastBoard::BOARDSQ;

        for(int idx = 0; idx < FastBoard::BOARDSQ; idx++) {

            auto rot_idx = rotate_nn_idx_table[rotation][idx];
            auto color = board.get_square(rot_idx);

            if (color != FastBoard::EMPTY) {
                if (color == to_move) {
                    me[idx] = 1;
                } else {
                    opp[idx] = 1;
                }
            }
        }
    }

    if (to_move == FastBoard::BLACK) {
        std::fill(black_to_move, black_to_move+FastBoard::BOARDSQ, 1);
    } else {
        std::fill(white_to_move, white_to_move+FastBoard::BOARDSQ, 1);
    }
}

void Network::gather_features(const GameState* state, std::vector<float> & planes) {
    planes.resize(INPUT_CHANNELS * FastBoard::BOARDSQ);
    gather_features(state, &planes[0], 0);
}

void Network::fill_input_plane_pair(const FastBoard& board,
                                    BoardPlane& black, BoardPlane& white) {
    for(int idx = 0; idx < FastBoard::BOARDSQ; idx++) {
        auto color = board.get_square(idx);
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
    planes.resize(INPUT_CHANNELS);
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
