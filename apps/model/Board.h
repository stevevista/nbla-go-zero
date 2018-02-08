#pragma once
#include <engine/tensor.h>

class GoBoard {
public:
    GoBoard() {
        reset();
    }

    int operator[](int i) const { return stones[i]; }

    void reset();
    void update_board(const int color, const int i);

    static bool validate_moves(const std::vector<int>& moves);

    template<typename ITERATOR>
	static void get_board_features(InputFeature& planes, ITERATOR begin, ITERATOR end) {
        
        for (auto& l : planes)
            l = false;

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
			if (h < 8 ) {
                // collect white, black occupation planes
                for (int pos=0; pos<361; pos++) {
                    auto color = b.stones[pos];
                    
                    if (color == next_to_move) {
                        planes[h][pos] = true;
                    } else if (color == -next_to_move) {
                        planes[8 + h][pos] = true;
                    }
                }
			}
		}
        
        if (next_to_move == 1)
            planes[16].set();
        else
            planes[17].set();
    }

private:
    void merge_strings(const int ip, const int aip);
    void remove_string(int i);

    int stones[361];
    int group_ids[361];
    int group_libs[361];
    int stone_next[361];
};
