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

#ifndef FASTSTATE_H_INCLUDED
#define FASTSTATE_H_INCLUDED

#include <cstddef>
#include <array>
#include <string>
#include <vector>

#include "FastBoard.h"

class FastState {
public:
    void init_game(float komi);
    void reset_game();

    void play_move(int vertex);

    bool is_move_legal(int color, int vertex);

    void set_komi(float komi);
    float get_komi() const;
    int get_passes() const;
    int get_to_move() const;
    void set_to_move(int tomove);
    void set_passes(int val);
    void increment_passes();

    float final_score() const;

    size_t get_movenum() const;
    int get_last_move() const;
    void display_state();

    bool superko(void) const;
    bool superko_move(const int color, const int i) const;

    FastBoard board;

    float m_komi;
    int m_passes;
    int m_komove;
    size_t m_movenum;
    int m_lastmove;
    std::vector<std::uint64_t> m_ko_hash_history;

protected:
    void play_move(int color, int vertex);
};

#endif
