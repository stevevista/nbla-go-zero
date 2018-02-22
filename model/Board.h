#pragma once

#include <array>
#include <vector>
#include <iostream>
#include <bitset>
#include "../leela/size_info.h"

extern std::array<std::array<int, 361>, 8> rotate_nn_idx_table;

using BoardPlane = std::bitset<361>;
using InputFeature = std::vector<BoardPlane>;


class GoBoard {
public:
    static void init_board();

    GoBoard() {
        reset();
    }

    int operator[](int i) const { return stones[i]; }

    void reset();
    void update_board(const int color, const int i);

    static bool validate_moves(const std::vector<int>& moves);

    template<typename ITERATOR>
	static void gather_features(InputFeature& planes, ITERATOR begin, ITERATOR end) {
        
        planes.resize(zero::input_channels);

        GoBoard b;

		int tomove = 1;
        const int maxcount = std::distance(begin, end);
        const int next_to_move = (maxcount%2 == 0) ? 1 : -1;
		
		auto it = begin;
		for (int i=0; i< maxcount; i++) {
                
            int idx = *it++;

            if (idx >= 0)
				b.update_board(tomove, idx);
			tomove = -tomove;
		
			// copy history
			int h = maxcount -i - 1;
			if (h < zero::input_history ) {
                // collect white, black occupation planes
                for (int pos=0; pos<361; pos++) {
                    auto color = b.stones[pos];
                    
                    if (color == next_to_move) {
                        planes[h][pos] = true;
                    } else if (color == -next_to_move) {
                        planes[zero::input_history + h][pos] = true;
                    }
                }
			}
		}
        
        if (next_to_move == 1)
            planes[zero::input_channels - 2].set();
        else
            planes[zero::input_channels - 1].set();
    }

private:
    void merge_strings(const int ip, const int aip);
    void remove_string(int i);

    int stones[361];
    int group_ids[361];
    int group_libs[361];
    int stone_next[361];
};
