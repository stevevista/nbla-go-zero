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
#include "../nn.h"

using namespace Utils;

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

Network::Netresult Network::get_scored_moves(
    const GameState * state, Ensemble ensemble, int rotation, bool skip_cache) {
    Netresult result;

    NNPlanes planes;
    gather_features(state, planes);

    // See if we already have this in the cache.
    if (!skip_cache) {
        if (NNCache::get_NNCache().lookup(planes, result)) {
          return result;
        }
    }

    if (ensemble == DIRECT) {
        assert(rotation >= 0 && rotation <= 7);
        result = get_scored_moves_internal(state, planes, rotation);
    } else {
        assert(ensemble == RANDOM_ROTATION);
        assert(rotation == -1);
        auto rand_rot = Random::get_Rng().randfix<8>();
        result = get_scored_moves_internal(state, planes, rand_rot);
    }

    // Insert result into cache.
    NNCache::get_NNCache().insert(planes, result);

    return result;
}

Network::Netresult Network::get_scored_moves_internal(
    const GameState * state, const NNPlanes & planes, int rotation) {

    assert(rotation >= 0 && rotation <= 7);
    assert(INPUT_CHANNELS == planes.size());
        
    // Data layout is input_data[(c * height + h) * width + w]
    NNPlanes inputs;
    for (int c = 0; c < INPUT_CHANNELS; ++c) {
        for (int idx = 0; idx < 361; ++idx) {
            auto rot_idx = rotate_nn_idx_table[rotation][idx];
            inputs[c][idx] = planes[c][rot_idx];
        }
    }
        
    std::lock_guard<std::mutex> lock(mtx_);
    std::vector<scored_node> result;
 
    auto out = zero_net->predict(inputs, cfg_softmax_temp);
    float winrate_out = out.second;
    auto outputs = out.first;

    for (auto idx = size_t{0}; idx < outputs.size(); idx++) {
            if (idx < 19*19) {
                auto val = outputs[idx];
                auto rot_idx = rotate_nn_idx_table[rotation][idx];
                auto x = rot_idx % 19;
                auto y = rot_idx / 19;
                auto rot_vtx = state->board.get_vertex(x, y);
                if (state->board.get_square(rot_vtx) == FastBoard::EMPTY) {
                    result.emplace_back(val, rot_vtx);
                }
            } else {
                result.emplace_back(outputs[idx], FastBoard::PASS);
            }
    }
    

    // Sigmoid
    float winrate_sig = (1.0f + winrate_out) / 2.0f;
        
    return std::make_pair(result, winrate_sig);
}


void Network::show_heatmap(const FastState * state, Netresult& result, bool topmoves) {
    auto moves = result.first;
    std::vector<std::string> display_map;
    std::string line;

    for (unsigned int y = 0; y < 19; y++) {
        for (unsigned int x = 0; x < 19; x++) {
            int vtx = state->board.get_vertex(x, y);

            auto item = std::find_if(moves.cbegin(), moves.cend(),
                [&vtx](scored_node const& test_item) {
                return test_item.second == vtx;
            });

            auto score = 0.0f;
            // Non-empty squares won't be scored
            if (item != moves.end()) {
                score = item->first;
                assert(vtx == item->second);
            }

            line +=boost::format("%3d ", int(score * 1000));
            if (x == 18) {
                display_map.push_back(line);
                line.clear();
            }
        }
    }

    for (int i = display_map.size() - 1; i >= 0; --i) {
        myprintf("%s\n", display_map[i].c_str());
    }
    assert(result.first.back().second == FastBoard::PASS);
    auto pass_score = int(result.first.back().first * 1000);
    myprintf("pass: %d\n", pass_score);
    myprintf("winrate: %f\n", result.second);

    if (topmoves) {
        std::stable_sort(rbegin(moves), rend(moves));

        auto cum = 0.0f;
        size_t tried = 0;
        while (cum < 0.85f && tried < moves.size()) {
            if (moves[tried].first < 0.01f) break;
            myprintf("%1.3f (%s)\n",
                    moves[tried].first,
                    state->board.move_to_text(moves[tried].second).c_str());
            cum += moves[tried].first;
            tried++;
        }
    }
}



void Network::fill_input_plane_pair(const FullBoard& board,
    BoardPlane& black, BoardPlane& white) {
    auto idx = 0;
    for (int j = 0; j < 19; j++) {
        for(int i = 0; i < 19; i++) {
            int vtx = board.get_vertex(i, j);
            auto color = board.get_square(vtx);
            if (color != FastBoard::EMPTY) {
                if (color == FastBoard::BLACK) {
                    black[idx] = true;
                } else {
                    white[idx] = true;
                }
            }
            idx++;
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



