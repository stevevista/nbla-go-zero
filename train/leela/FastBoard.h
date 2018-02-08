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
#include "Random.h"

#include <array>
#include <queue>
#include <string>
#include <utility>
#include <vector>




class FastBoard {
    friend class FastState;
    friend class GameState;
public:

    static constexpr int BOARDSIZE = 19;
    static constexpr int BOARDSQ = BOARDSIZE*BOARDSIZE;

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

    using StoneMap = std::array<square_t, BOARDSQ>;



    square_t get_square(int x, int y) const;
    square_t get_square(int vertex) const ;

    bool is_eye_shape(int i, int color) const;
    bool is_eye(int i, int color) const;
    bool is_suicide(int i, int color) const;

    float area_score(float komi) const;

    std::string move_to_text(int move) const;
    std::string move_to_text_sgf(int move) const;

    void reset_board(int size);
    void display_board(int lastmove = -1);

    static bool starpoint(int size, int point);
    static bool starpoint(int size, int x, int y);

    
    int update_board(const int color, const int i);

    
    float playout(int to_move, int ko, int rounds, float komi) const;

    std::uint64_t get_ko_hash(void) const;

    int verify(int i, int visited_libs[]);
    void verify_board();
    void auto_test();

    bool fast_test_move(const int color, const int i, std::uint64_t& ko_hash) const;

protected:
    int random_move(int color, int& ko);
    void roll_to_end(int color, int ko, int& blacks, int& whites) const;


    std::array<square_t, BOARDSQ>          m_square;      /* board contents */
    std::array<unsigned short, BOARDSQ>    m_next;        /* next stone in string */
    std::array<unsigned short, BOARDSQ>    m_parent;      /* parent node of string */
    std::array<unsigned short, BOARDSQ>    m_libs;        /* liberties per string parent */
    std::array<unsigned short, BOARDSQ>    m_stones;      /* stones per string parent */
    std::uint64_t m_ko_hash;

    int calc_reach_color(int color) const;

    int remove_string(int i);
    void merge_strings(const int ip, const int aip);
    void remove_neighbour(const int i, const int color);

};



class Zobrist {
public:
    static std::array<std::array<std::uint64_t, FastBoard::BOARDSQ>,     3> zobrist;
    static void init_zobrist(Random& rng);
};


#endif
