#pragma once

#include <array>
#include <vector>
#include <iostream>
#include <bitset>
#include "../leela/size_info.h"

extern std::array<std::array<int, 361>, 8> rotate_nn_idx_table;

class GoBoard {
public:
    static void init_board();

    GoBoard() {
        reset();
    }

    void reset();
    void update_board(const int color, const int i, std::vector<int>& removed);

    static bool generate_move_seqs(const std::vector<int>& moves, std::vector<short>& seqs);

private:
    void merge_strings(const int ip, const int aip);
    void remove_string(int i, std::vector<int>& removed);

    int stones[361];
    int group_ids[361];
    int group_libs[361];
    int stone_next[361];
};
