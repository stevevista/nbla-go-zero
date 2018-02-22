#pragma once

namespace zero {

constexpr int board_size = 19;
constexpr int board_count = board_size*board_size;
constexpr int board_moves = board_count + 1;

static constexpr int input_history = 8;
static constexpr int input_channels = 2 * input_history + 2;

constexpr int RESIDUAL_FILTERS = 192;
constexpr int RESIDUAL_BLOCKS = 13;



}
