#include "Board.h"
#include <stdexcept>
#include <cassert>



static std::array<std::vector<int>, 361> NEIGHBORS;

// Rotation helper
std::array<std::array<int, 361>, 8> rotate_nn_idx_table;

int rotate_nn_idx(const int vertex, int symmetry) {

    assert(vertex >= 0 && vertex < 361);
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

    return newvtx;
}

void GoBoard::init_board() {
    for (int y=0; y<19; y++) {
        for (int x=0; x<19; x++) {
            auto& n = NEIGHBORS[y*19 + x];

            if (y > 0) n.emplace_back((y-1)*19 + x);
            if (y < 19-1) n.emplace_back((y+1)*19 + x);
            if (x > 0) n.emplace_back(y*19 + x - 1);
            if (x < 19-1) n.emplace_back(y*19 + x + 1);
        }
    }

    // Prepare rotation table
    for(auto s = 0; s < 8; s++) {
        for(auto v = 0; v < 19 * 19; v++) {
            rotate_nn_idx_table[s][v] = rotate_nn_idx(v, s);
        }
    }
}


class NeighborVistor {
    int nbr_pars[4];
    int nbr_par_cnt = 0;
public:
    bool visited(int pos) {
        for (int i = 0; i < nbr_par_cnt; i++) {
            if (nbr_pars[i] == pos) {
                return true;
            }
        }
        nbr_pars[nbr_par_cnt++] = pos;
        return false;
    }
};

void GoBoard::reset() {
    
    for (int i = 0; i < 361; i++) {
        stones[i] = 0;
    }
}

void GoBoard::update_board(const int color, const int i, std::vector<int>& removed) {

    if (i <0 || i >= 361 || stones[i]) {
        std::cout << i << std::endl;
        std::cout << stones[i] << std::endl;
        throw std::runtime_error("update board error");
    }

    stones[i] = color;
    stone_next[i] = i;
    group_ids[i] = i;

    int libs = 0;
    NeighborVistor vistor;

    for (auto ai : NEIGHBORS[i]) {

        if (stones[ai] == 0)
            libs++;
        else {
            int g = group_ids[ai];
            if (!vistor.visited(g)) {
                group_libs[g]--;
            }
        }
    }

    group_libs[i] = libs;

    for (auto ai : NEIGHBORS[i]) {

        if (stones[ai] == -color) {
            if (group_libs[group_ids[ai]] == 0) {
                remove_string(ai, removed);
            }
        } else if (stones[ai] == color) {
            int ip = group_ids[i];
            int aip = group_ids[ai];

            if (ip != aip) {
                merge_strings(aip, ip);
            }
        }
    }

    // check whether we still live (i.e. detect suicide)
    if (group_libs[group_ids[i]] == 0) {
        remove_string(group_ids[i], removed);
    }
}

void GoBoard::remove_string(int i, std::vector<int>& removed) {

    int pos = i;

    do {
        stones[pos]  = 0;
        removed.emplace_back(pos);

        NeighborVistor vistor;
        
        for (auto ai : NEIGHBORS[pos]) {

            if (!stones[ai]) continue;

            if (!vistor.visited(group_ids[ai])) {
                group_libs[group_ids[ai]]++;
            }
        }

        pos = stone_next[pos];
    } while (pos != i);
}

void GoBoard::merge_strings(const int ip, const int aip) {

    /* loop over stones, update parents */
    int newpos = aip;

    do {
        // check if this stone has a liberty
        for (auto ai : NEIGHBORS[newpos]) {

            // for each liberty, check if it is not shared
            if (stones[ai] == 0) {
                // find liberty neighbors
                bool found = false;
                for (auto aai : NEIGHBORS[ai]) {

                    if (!stones[aai]) continue;

                    // friendly string shouldn't be ip
                    // ip can also be an aip that has been marked
                    if (group_ids[aai] == ip) {
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    group_libs[ip]++;
                }
            }
        }

        group_ids[newpos] = ip;
        newpos = stone_next[newpos];
    } while (newpos != aip);

    /* merge stings */
    int tmp = stone_next[aip];
    stone_next[aip] = stone_next[ip];
    stone_next[ip] = tmp;
}


bool GoBoard::generate_move_seqs(const std::vector<int>& moves, std::vector<short>& seqs) {

    GoBoard b;
    int color = 1;

    seqs.clear();

    for (auto idx : moves) {

        if (idx < 0 || idx > 361) {
            return false; // invalid move
        }

        std::vector<int> removed;
        short sign_idx = (short)(idx + 1);
        
        if (idx == 361) {
            // pass or resign
        } else {
            if (b.stones[idx] != 0) {
                return false; // invalid move
            }

            b.update_board(color, idx, removed);
        }

        if (removed.size()) 
            sign_idx = -sign_idx;

        seqs.emplace_back(sign_idx);
        if (removed.size()) {
            seqs.emplace_back((short)(removed.size()));
            for (auto r : removed)
                seqs.emplace_back((short)r);
        }
            
        color = -color;
    }

    return true;
}
