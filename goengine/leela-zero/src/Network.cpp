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
#include <boost/utility.hpp>
#include <boost/format.hpp>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#include "UCTNode.h"

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GameState.h"
#include "GTP.h"
#include "Im2Col.h"
#include "NNCache.h"
#include "Random.h"
#include "ThreadPool.h"
#include "Timing.h"
#include "Utils.h"
#include "nn.h"

using namespace Utils;

static std::mutex mtx_;
std::shared_ptr<zero_model> zero_net;


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

void Network::process_bn_var(std::vector<float>& weights, const float epsilon) {
    for(auto&& w : weights) {
        w = 1.0f / std::sqrt(w + epsilon);
    }
}

std::vector<float> Network::winograd_transform_f(const std::vector<float>& f,
                                                 const int outputs,
                                                 const int channels) {
    // F(2x2, 3x3) Winograd filter transformation
    // transpose(G.dot(f).dot(G.transpose()))
    // U matrix is transposed for better memory layout in SGEMM
    auto U = std::vector<float>(WINOGRAD_TILE * outputs * channels);
    auto G = std::array<float, WINOGRAD_TILE>{ 1.0,  0.0,  0.0,
                                               0.5,  0.5,  0.5,
                                               0.5, -0.5,  0.5,
                                               0.0,  0.0,  1.0};
    auto temp = std::array<float, 12>{};

    for (auto o = 0; o < outputs; o++) {
        for (auto c = 0; c < channels; c++) {
            for (auto i = 0; i < 4; i++){
                for (auto j = 0; j < 3; j++) {
                    auto acc = 0.0f;
                    for (auto k = 0; k < 3; k++) {
                        acc += G[i*3 + k] * f[o*channels*9 + c*9 + k*3 + j];
                    }
                    temp[i*3 + j] = acc;
                }
            }

            for (auto xi = 0; xi < 4; xi++) {
                for (auto nu = 0; nu < 4; nu++) {
                    auto acc = 0.0f;
                    for (int k = 0; k < 3; k++) {
                        acc += temp[xi*3 + k] * G[nu*3 + k];
                    }
                    U[xi * (4 * outputs * channels)
                      + nu * (outputs * channels)
                      + c * outputs
                      + o] = acc;
                }
            }
        }
    }

    return U;
}

std::vector<float> Network::zeropad_U(const std::vector<float>& U,
                                      const int outputs, const int channels,
                                      const int outputs_pad,
                                      const int channels_pad) {
    // Fill with zeroes
    auto Upad = std::vector<float>(WINOGRAD_TILE * outputs_pad * channels_pad);

    for(auto o = 0; o < outputs; o++) {
        for(auto c = 0; c < channels; c++) {
            for(auto xi = 0; xi < WINOGRAD_ALPHA; xi++){
                for(auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                    Upad[xi * (WINOGRAD_ALPHA * outputs_pad * channels_pad)
                         + nu * (outputs_pad * channels_pad)
                         + c * outputs_pad +
                          o] =
                    U[xi * (WINOGRAD_ALPHA * outputs * channels)
                      + nu * (outputs * channels)
                      + c * outputs
                      + o];
                }
            }
        }
    }

    return Upad;
}


std::pair<int, int> Network::load_network_file(std::string filename) {
    if (!zero_net)
        zero_net = std::make_shared<zero_model>();
    if  (!zero_net->load_weights(filename))
        return {0, 0};

    switch(zero_net->type()) {
        case 1:
            return {256, 19};
        case 2:
            return {192, 6};
        default:
            return {256, 13};  //darknet
    }
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
        exit(EXIT_FAILURE);
    }

    myprintf("Initializing NN backend.\n");

    if (cfg_tune_only) {
        exit(EXIT_SUCCESS);
    }
}

void Network::softmax(const std::vector<float>& input,
                      std::vector<float>& output,
                      float temperature) {
    assert(&input != &output);

    auto alpha = *std::max_element(begin(input),
                                   begin(input) + output.size());
    alpha /= temperature;

    auto denom = 0.0f;
    auto helper = std::vector<float>(output.size());
    for (auto i = size_t{0}; i < output.size(); i++) {
        auto val   = std::exp((input[i]/temperature) - alpha);
        helper[i]  = val;
        denom     += val;
    }
    for (auto i = size_t{0}; i < output.size(); i++) {
        output[i] = helper[i] / denom;
    }
}

Network::Netresult Network::get_scored_moves(
    const GameState* state, Ensemble ensemble, int rotation, bool skip_cache) {
    Netresult result;
    if (state->board.get_boardsize() != 19) {
        return result;
    }

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
    const GameState* state, NNPlanes & planes, int rotation) {
    assert(rotation >= 0 && rotation <= 7);
    assert(INPUT_CHANNELS == planes.size());
    constexpr int width = 19;
    constexpr int height = 19;

    NNPlanes input_data;

    for (int c = 0; c < INPUT_CHANNELS; ++c) {
        BoardPlane ft;
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                auto rot_idx = rotate_nn_idx_table[rotation][h * 19 + w];
                ft[h * 19 + w] = planes[c][rot_idx];
            }
        }
        input_data.emplace_back(ft);
    }

    std::pair<std::vector<float>, float>  out;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        out = zero_net->predict(input_data, cfg_softmax_temp);
    }
    float winrate_out = out.second;
    auto& outputs = out.first;

    // Sigmoid
    auto winrate_sig = (1.0f + winrate_out) / 2.0f;

    std::vector<scored_node> result;
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
