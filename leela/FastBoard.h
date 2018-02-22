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

#ifndef FASTBOARD_H_INCLUDED
#define FASTBOARD_H_INCLUDED

#include "config.h"

#include <array>
#include <queue>
#include <string>
#include <utility>
#include <vector>

class FastBoard {
    friend class FastState;
public:
    /*
        infinite score
    */
    static constexpr int BIG = 10000000;

    /*
        vertex of a pass
    */
    static constexpr int PASS   = -1;
    /*
        vertex of a "resign move"
    */
    static constexpr int RESIGN = -2;

    /*
        possible contents of a square
    */
    enum square_t : char {
        BLACK = 0, WHITE = 1, EMPTY = 2
    };

    static constexpr int BOARDSIZE = 19;
    static constexpr int BOARDSQ = BOARDSIZE * BOARDSIZE;

    static std::array<std::vector<int>, BOARDSQ> NEIGHBORS;
    static std::array<std::vector<int>, BOARDSQ> DIAGS;

    static void init_board();

    /*
        move generation types
    */
    using movescore_t = std::pair<int, float>;
    using scoredmoves_t = std::vector<movescore_t>;

    square_t get_square(int x, int y) const;
    square_t get_square(int vertex) const ;

    bool is_suicide(int i, int color) const;
    bool is_eye(const int color, const int vtx) const;

    float area_score(float komi) const;

    int get_prisoners(int side) const;
    bool black_to_move() const;
    bool white_to_move() const;
    int get_to_move() const;
    void set_to_move(int color);

    static std::string move_to_text(int move);
    std::string move_to_text_sgf(int move) const;
    std::string get_stone_list() const;
    std::string get_string(int vertex) const;
    static int text_to_move(const std::string& vertex);

    void reset_board();
    void display_board(int lastmove = -1);

    static bool starpoint(int size, int point);
    static bool starpoint(int size, int x, int y);

    int remove_string(int i);
    int update_board(const int color, const int i);

    std::uint64_t test_update_ko_hash(const int color, const int i) const;

    std::uint64_t calc_hash(int komove = 0);
    std::uint64_t calc_ko_hash(void);
    std::uint64_t get_hash(void) const;
    std::uint64_t get_ko_hash(void) const;
    
protected:
    std::array<square_t, BOARDSQ>          m_square;      /* board contents */
    std::array<unsigned short, BOARDSQ>    m_next;        /* next stone in string */
    std::array<unsigned short, BOARDSQ>    m_parent;      /* parent node of string */
    std::array<unsigned short, BOARDSQ>    m_libs;        /* liberties per string parent */
    std::array<unsigned short, BOARDSQ>    m_stones;      /* stones per string parent */
    std::array<int, 2>                     m_prisoners;   /* prisoners per color */

    int m_tomove;
    std::uint64_t m_hash;
    std::uint64_t m_ko_hash;

    int calc_reach_color(int color) const;

    void merge_strings(const int ip, const int aip);
    void print_columns();
};

#endif
