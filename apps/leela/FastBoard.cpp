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

#include "FastBoard.h"

#include <cassert>
#include <array>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#include "Utils.h"

using namespace Utils;


constexpr int FastBoard::PASS;
constexpr int FastBoard::RESIGN;

std::array<std::array<std::uint64_t, FastBoard::BOARDSQ>,     3> Zobrist::zobrist;

void Zobrist::init_zobrist(Random& rng) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < FastBoard::BOARDSQ; j++) {
            Zobrist::zobrist[i][j]  = ((std::uint64_t)rng.randuint32()) << 32;
            Zobrist::zobrist[i][j] ^= (std::uint64_t)rng.randuint32();
        }
    }
}




constexpr int NEIGHBOR_N[361] = { 
   -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1, 
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18, 
   19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37, 
   38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56, 
   57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75, 
   76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94, 
   95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 
  114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 
  133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 
  152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 
  171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 
  190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 
  209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 
  228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 
  247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 
  266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 
  285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 
  304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 
  323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341,
};

constexpr int NEIGHBOR_S[361] = { 
    19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37, 
    38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56, 
    57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75, 
    76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94, 
    95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 
   114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 
   133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 
   152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 
   171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 
   190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 
   209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 
   228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 
   247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 
   266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 
   285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 
   304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 
   323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 
   342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 
   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};

constexpr int NEIGHBOR_W[361] = { 
    -1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17, 
    -1,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36, 
    -1,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55, 
    -1,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74, 
    -1,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93, 
    -1,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 
    -1, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 
    -1, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 
    -1, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 
    -1, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 
    -1, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 
    -1, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 
    -1, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 
    -1, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 
    -1, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 
    -1, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 
    -1, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 
    -1, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 
    -1, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359,
};

constexpr int NEIGHBOR_E[361] = { 
    1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18, -1, 
    20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37, -1, 
    39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56, -1, 
    58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75, -1, 
    77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94, -1, 
    96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, -1, 
   115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, -1, 
   134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, -1, 
   153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, -1, 
   172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, -1, 
   191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, -1, 
   210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, -1, 
   229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, -1, 
   248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, -1, 
   267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, -1, 
   286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, -1, 
   305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, -1, 
   324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, -1, 
   343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, -1,
};

constexpr const int* NEIGHBORS[4] = {
    NEIGHBOR_N,
    NEIGHBOR_S,
    NEIGHBOR_W,
    NEIGHBOR_E
};


FastBoard::square_t FastBoard::get_square(int vertex) const {
    assert(vertex >= 0 && vertex < BOARDSQ);
    return m_square[vertex];
}

FastBoard::square_t FastBoard::get_square(int x, int y) const {
    return get_square(y*BOARDSIZE+x);
}

void FastBoard::reset_board(int size) {

    std::fill(m_square.begin(), m_square.end(), EMPTY);

    auto res = std::uint64_t{0x1234567887654321ULL};

    for (int i = 0; i < BOARDSQ; i++) {
        res ^= Zobrist::zobrist[EMPTY][i];
    }

    m_ko_hash = res;
}

// only use for score calc 
// NOT use for move legalize
bool FastBoard::is_eye_shape(int i, int color) const {

    if (m_square[i] != EMPTY)
        return false;

    for (int k = 0; k < 4; k++) {
        int ai = NEIGHBORS[k][i];
        if (ai < 0) continue;
        if (color != m_square[ai])
            return false;
    }

    return true;
}

bool FastBoard::is_eye(int i, int color) const {

    if (!is_eye_shape(i, color))
        return false;

    int edges = 0;
    int diags = 0;
    if (NEIGHBOR_N[i] < 0 || NEIGHBOR_W[i] < 0) edges++;
    else {
        if (m_square[i-1-BOARDSIZE] == !color) diags++;
    }

    if (NEIGHBOR_N[i] < 0 || NEIGHBOR_E[i] < 0) edges++;
    else {
        if (m_square[i+1-BOARDSIZE] == !color) diags++;
    }

    if (NEIGHBOR_S[i] < 0 || NEIGHBOR_W[i] < 0) edges++;
    else {
        if (m_square[i-1+BOARDSIZE] == !color) diags++;
    }

    if (NEIGHBOR_S[i] < 0 || NEIGHBOR_E[i] < 0) edges++;
    else {
        if (m_square[i+1+BOARDSIZE] == !color) diags++;
    }    

    if (edges == 0 && diags > 1) {
        return false;
    }

    if (edges > 0 && diags > 0) {
        return false;
    }

    return true;
}



bool FastBoard::is_suicide(int i, int color) const {

    // If we get here, we played in a "hole" surrounded by stones
    for (auto k = 0; k < 4; k++) {
        auto ai = NEIGHBORS[k][i];
        if (ai < 0) continue;

        if (m_square[ai] == EMPTY) {
            // If there are liberties next to us, it is never suicide
            return false;
        } else if (m_square[ai] == color) {
            auto libs = m_libs[m_parent[ai]];
            if (libs > 1) {
                // connecting to live group = not suicide
                return false;
            }
        } else if (m_square[ai] == !color) {
            auto libs = m_libs[m_parent[ai]];
            if (libs <= 1) {
                // killing neighbour = not suicide
                return false;
            }
        }
    }

    // We played in a hole, friendlies had one liberty at most and
    // we did not kill anything. So we killed ourselves.
    return true;
}


void FastBoard::remove_neighbour(const int vtx, const int color) {
    assert(color == WHITE || color == BLACK || color == EMPTY);

    std::array<int, 4> nbr_pars;
    int nbr_par_cnt = 0;

    for (int k = 0; k < 4; k++) {
        int ai = NEIGHBORS[k][vtx];
        if (ai < 0 || m_square[ai] == EMPTY) continue;

        bool found = false;
        for (int i = 0; i < nbr_par_cnt; i++) {
            if (nbr_pars[i] == m_parent[ai]) {
                found = true;
                break;
            }
        }
        if (!found) {
            m_libs[m_parent[ai]]++;
            nbr_pars[nbr_par_cnt++] = m_parent[ai];
        }
    }
}

int FastBoard::calc_reach_color(int color) const {
    auto reachable = 0;
    auto bd = std::vector<bool>(BOARDSQ, false);
    auto open = std::queue<int>();
    for (auto vertex = 0; vertex < BOARDSQ; vertex++) {
            if (m_square[vertex] == color) {
                reachable++;
                bd[vertex] = true;
                open.push(vertex);
            }
    }
    while (!open.empty()) {
        /* colored field, spread */
        auto vertex = open.front();
        open.pop();

        for (auto k = 0; k < 4; k++) {
            auto neighbor = NEIGHBORS[k][vertex];
            if (neighbor < 0) continue;

            if (!bd[neighbor] && m_square[neighbor] == EMPTY) {
                reachable++;
                bd[neighbor] = true;
                open.push(neighbor);
            }
        }
    }
    return reachable;
}

// Needed for scoring passed out games not in MC playouts
float FastBoard::area_score(float komi) const {
    auto white = calc_reach_color(WHITE);
    auto black = calc_reach_color(BLACK);
    return black - white - komi;
}

void FastBoard::display_board(int lastmove) {

    int boardsize = BOARDSIZE;

    myprintf("\n   ");
    for (int i = 0; i < boardsize; i++) {
        if (i < 25) {
            myprintf("%c ", (('a' + i < 'i') ? 'a' + i : 'a' + i + 1));
        } else {
            myprintf("%c ", (('A' + (i-25) < 'I') ? 'A' + (i-25) : 'A' + (i-25) + 1));
        }
    }
    myprintf("\n");
    for (int j = BOARDSIZE-1; j >= 0; j--) {
        myprintf("%2d", j+1);
        if (lastmove == j * BOARDSIZE)
            myprintf("(");
        else
            myprintf(" ");
        for (int i = 0; i < BOARDSIZE; i++) {
            if (get_square(i,j) == WHITE) {
                myprintf("O");
            } else if (get_square(i,j) == BLACK)  {
                myprintf("X");
            } else if (starpoint(boardsize, i, j)) {
                myprintf("+");
            } else {
                myprintf(".");
            }
            if (lastmove == j * BOARDSIZE + i) myprintf(")");
            else if (i != boardsize-1 && lastmove == j * BOARDSIZE + i +1) myprintf("(");
            else myprintf(" ");
        }
        myprintf("%2d\n", j+1);
    }
    myprintf("   ");
    for (int i = 0; i < boardsize; i++) {
         if (i < 25) {
            myprintf("%c ", (('a' + i < 'i') ? 'a' + i : 'a' + i + 1));
        } else {
            myprintf("%c ", (('A' + (i-25) < 'I') ? 'A' + (i-25) : 'A' + (i-25) + 1));
        }
    }
    myprintf("\n\n");

    myprintf("Ko-Hash: %llX\n\n", get_ko_hash());
}

void FastBoard::merge_strings(const int ip, const int aip) {

    /* merge stones */
    m_stones[ip] += m_stones[aip];

    /* loop over stones, update parents */
    int newpos = aip;

    do {
        // check if this stone has a liberty
        for (int k = 0; k < 4; k++) {
            int ai = NEIGHBORS[k][newpos];
            if (ai < 0) continue;

            // for each liberty, check if it is not shared
            if (m_square[ai] == EMPTY) {
                // find liberty neighbors
                bool found = false;
                for (int kk = 0; kk < 4; kk++) {
                    int aai = NEIGHBORS[kk][ai];
                    if (aai < 0 || m_square[aai] == EMPTY) continue;

                    // friendly string shouldn't be ip
                    // ip can also be an aip that has been marked
                    if (m_parent[aai] == ip) {
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    m_libs[ip]++;
                }
            }
        }

        m_parent[newpos] = ip;
        newpos = m_next[newpos];
    } while (newpos != aip);

    /* merge stings */
    std::swap(m_next[aip], m_next[ip]);
}

std::string FastBoard::move_to_text(int move) const {
    std::ostringstream result;

    int column = move % BOARDSIZE;
    int row = move / BOARDSIZE;

    assert(move == FastBoard::PASS || move == FastBoard::RESIGN || (row >= 0 && row < BOARDSIZE));
    assert(move == FastBoard::PASS || move == FastBoard::RESIGN || (column >= 0 && column < BOARDSIZE));

    if (move >= 0 && move < BOARDSQ) {
        result << static_cast<char>(column < 8 ? 'A' + column : 'A' + column + 1);
        result << (row + 1);
    } else if (move == FastBoard::PASS) {
        result << "pass";
    } else if (move == FastBoard::RESIGN) {
        result << "resign";
    } else {
        result << "error";
    }

    return result.str();
}

std::string FastBoard::move_to_text_sgf(int move) const {
    std::ostringstream result;

    int column = move % BOARDSIZE;
    int row = move / BOARDSIZE;

    assert(move == FastBoard::PASS || move == FastBoard::RESIGN || (row >= 0 && row < BOARDSIZE));
    assert(move == FastBoard::PASS || move == FastBoard::RESIGN || (column >= 0 && column < BOARDSIZE));

    // SGF inverts rows
    row = BOARDSIZE - row - 1;

    if (move >= 0 && move < BOARDSQ) {
        if (column <= 25) {
            result << static_cast<char>('a' + column);
        } else {
            result << static_cast<char>('A' + column - 26);
        }
        if (row <= 25) {
            result << static_cast<char>('a' + row);
        } else {
            result << static_cast<char>('A' + row - 26);
        }
    } else if (move == FastBoard::PASS) {
        result << "tt";
    } else if (move == FastBoard::RESIGN) {
        result << "tt";
    } else {
        result << "error";
    }

    return result.str();
}

bool FastBoard::starpoint(int size, int point) {
    int stars[3];
    int points[2];
    int hits = 0;

    if (size % 2 == 0 || size < 9) {
        return false;
    }

    stars[0] = size >= 13 ? 3 : 2;
    stars[1] = size / 2;
    stars[2] = size - 1 - stars[0];

    points[0] = point / size;
    points[1] = point % size;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (points[i] == stars[j]) {
                hits++;
            }
        }
    }

    return hits >= 2;
}

bool FastBoard::starpoint(int size, int x, int y) {
    return starpoint(size, y * size + x);
}

int FastBoard::remove_string(int i) {
    int pos = i;
    int removed = 0;
    int color = m_square[i];

    do {
        m_ko_hash ^= Zobrist::zobrist[m_square[pos]][pos];

        m_square[pos] = EMPTY;
        remove_neighbour(pos, color);

        m_ko_hash ^= Zobrist::zobrist[m_square[pos]][pos];

        removed++;
        pos = m_next[pos];
    } while (pos != i);

    return removed;
}

std::uint64_t FastBoard::get_ko_hash(void) const {
    return m_ko_hash;
}

int FastBoard::update_board(const int color, const int i) {
    assert(m_square[i] == EMPTY);
    assert(color == WHITE || color == BLACK);

    m_ko_hash ^= Zobrist::zobrist[m_square[i]][i];

    m_square[i] = (square_t)color;
    m_next[i] = i;
    m_parent[i] = i;
    m_stones[i] = 1;

    int orgin_libs = 0;
    std::array<int, 4> nbr_pars;
    int nbr_par_cnt = 0;

    for (int k = 0; k < 4; k++) {
        int ai = NEIGHBORS[k][i];
        if (ai < 0) continue;

        if (m_square[ai] == EMPTY) {
            orgin_libs++;
        } else {
            /* update neighbor liberties (they all lose 1) */
            bool found = false;
            for (int i = 0; i < nbr_par_cnt; i++) {
                if (nbr_pars[i] == m_parent[ai]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                m_libs[m_parent[ai]]--;
                nbr_pars[nbr_par_cnt++] = m_parent[ai];
            }
        }
    }

    m_libs[i] = orgin_libs;

    m_ko_hash ^= Zobrist::zobrist[m_square[i]][i];

    auto captured_stones = 0;
    int captured_sq;

    for (int k = 0; k < 4; k++) {
        int ai = NEIGHBORS[k][i];
        if (ai < 0) continue;

        if (m_square[ai] == !color) {
            if (m_libs[m_parent[ai]] <= 0) {
                int this_captured = remove_string(ai);
                captured_sq = ai;
                captured_stones += this_captured;
            }
        } else if (m_square[ai] == color) {
            int ip = m_parent[i];
            int aip = m_parent[ai];

            if (ip != aip) {
                if (m_stones[ip] >= m_stones[aip]) {
                    merge_strings(ip, aip);
                } else {
                    merge_strings(aip, ip);
                }
            }
        }
    }

    /* check whether we still live (i.e. detect suicide) */
    if (m_libs[m_parent[i]] == 0) {
        assert(captured_stones == 0);
        remove_string(i);
    }

    /* check for possible simple ko */
    if (captured_stones == 1 && orgin_libs == 0) {

        return captured_sq;
    }

    // No ko
    return -1;
}



int FastBoard::verify(int i, int visited_libs[]) {

    int color = m_square[i];
    bool visited[361];
    std::vector<int> stack;
    int libs = 0;

    std::fill(&visited[0], &visited[0]+361, false);
    stack.push_back(i);

    while (!stack.empty()) {

        bool move_on = false;
        int pos = stack.back();

        for (int k = 0; k < 4; k++) {
            int ai = NEIGHBORS[k][pos];
            if (ai < 0 || visited[ai]) continue;

            visited[ai] = true;

            if (m_square[ai] == EMPTY) {
                libs++;
            } else if (m_square[ai] == color) {
                if (visited_libs[ai] >=0)
                    return visited_libs[ai];

                move_on = true;
                stack.push_back(ai);
                break;
            }
        }   

        if (move_on)
            continue;

        stack.erase(stack.end()-1);
    }

    return libs;
}


void FastBoard::verify_board() {

    int visited_libs[361];
    std::fill(&visited_libs[0], &visited_libs[0]+361, -1);

    for (int i=0; i<361; i++) {
        if (m_square[i] != EMPTY) {
            int libs = verify(i, visited_libs);
            visited_libs[i] = libs;
            if (libs != m_libs[m_parent[i]]) {
                std::cout << ">>>> " << (i%19+1) << "," << (i/19+1) << std::endl;
                std::cout << "real libs: " << libs << std::endl;
                std::cout << "libs[" << m_parent[i] << "]: " << m_libs[m_parent[i]] << std::endl;
                display_board();
                throw std::runtime_error("board error");
            }
        }
    }
}



void FastBoard::auto_test() {

    for (int i=0; i<100; i++) {
        reset_board(19);
        int ko = -1;
        int prev = 0;
        int color = BLACK;
        int move_cnt = 0;

        while(true) {
            int move = random_move(color, ko);
            //std::cout << color << ": " << move << std::endl;
            
            if (move == -1 && prev == -1) {
                break;
            }

            color = !color;
            prev = move;
            move_cnt++;

            if (move_cnt > 400)
                break;
            
        }

        auto v1 = playout(BLACK, -1, 100, 0);
        auto v2 = area_score(0);

        if ((v1 > 0.5 && v2 <0) || (v1 < 0.5 && v2 > 0) ) {
            std::cout << v1 << std::endl;
            std::cout << v2 << std::endl;

            std::cout << "with moves " << move_cnt << std::endl;
            display_board();
            break;
        }

        
    }
    //display_board();
    std::cout << "test done! " << std::endl;
}




int FastBoard::random_move(int color, int& ko) {

    std::vector<int> candis;
    for (int i=0; i<361; i++) {
        if (m_square[i] == EMPTY && i != ko && !is_suicide(i, color) && !is_eye(i, color)) {
            candis.push_back(i);
        }
    }

    if (candis.size() == 0) {
        ko = -1;
        return -1;
    }

    int pick;
    if (candis.size() == 1) 
        pick = 0;
    else {
        pick = Random::get_Rng().randuint32(candis.size());
    }

    int move = candis[pick];
    ko = update_board(color, move);
    return move;
}

void FastBoard::roll_to_end(int color, int ko, int& blacks, int& whites) const {

    auto b = *this;
    int prev = 0;

    while(true) {
        int move = b.random_move(color, ko);
        if (move == -1 && prev == -1) {
            break;
        }

        color = !color;
        prev = move;
    }

    blacks = 0;
    whites = 0;
    for (int i=0; i< BOARDSQ; i++) {
        if (b.m_square[i] == BLACK || b.is_eye_shape(i, BLACK))
            blacks++;
        else 
            whites++;
    }
}

float FastBoard::playout(int to_move, int ko, int rounds, float komi) const {

    clock_t time=clock();

    int wins = 0;
    for (int i=0; i<rounds; i++) {
        int black = 0;
        int white = 0;
        roll_to_end(to_move, ko, black, white);
        float ret = black - white - komi;
        if (ret > 0) wins++;
    }

    //std::cout << "time: " << (float)(clock()-time)/CLOCKS_PER_SEC << std::endl;;

    return wins/(float)rounds;
}



bool FastBoard::fast_test_move(const int color, const int idx, std::uint64_t& ko_hash) const {

    if (m_square[idx] != EMPTY)
        return false;

    bool suicide = true;
    std::array<int, 4> kills;
    int kills_cnt = 0;

    for (int k = 0; k < 4; k++) {
        int ai = NEIGHBORS[k][idx];
        if (ai < 0) continue;

        if (m_square[ai] == EMPTY) {
            // If there are liberties next to us, it is never suicide
            suicide = false;
        } else if (m_square[ai] == color) {
            auto libs = m_libs[m_parent[ai]];
            if (libs > 1) {
                // connecting to live group = not suicide
                suicide = false;
            }
        } else if (m_square[ai] == !color) {

            /* update neighbor liberties (they all lose 1) */
            bool found = false;
            for (int i = 0; i < kills_cnt; i++) {
                if (kills[i] == m_parent[ai]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (m_libs[m_parent[ai]] == 1) {
                    kills[kills_cnt++] = m_parent[ai];
                    // killing neighbour = not suicide
                    suicide = false;
                }
            }
        }
    }

    if (suicide)
        return false;

    ko_hash = m_ko_hash;
    ko_hash ^= Zobrist::zobrist[EMPTY][idx];
    ko_hash ^= Zobrist::zobrist[color][idx];

    for (int i = 0; i < kills_cnt; i++) {
        int ip = kills[i];
        int pos = ip;

        do {
            ko_hash ^= Zobrist::zobrist[!color][pos];
            ko_hash ^= Zobrist::zobrist[EMPTY][pos];
            pos = m_next[pos];
        } while (pos != ip);
    }

    return true;
}

