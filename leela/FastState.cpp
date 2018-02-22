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
#include "FastState.h"

#include <algorithm>
#include <iterator>
#include <vector>

#include "FastBoard.h"
#include "Utils.h"
#include "Zobrist.h"

using namespace Utils;

void FastState::init_game(float komi) {
    board.reset_board();

    m_movenum = 0;

    m_komove = 0;
    m_lastmove = 0;
    m_komi = komi;
    m_passes = 0;

    m_ko_hash_history.clear();
    m_ko_hash_history.emplace_back(board.get_ko_hash());
}

void FastState::set_komi(float komi) {
    m_komi = komi;
}

void FastState::reset_game(void) {
    board.reset_board();

    m_movenum = 0;
    m_passes = 0;
    m_komove = 0;
    m_lastmove = 0;

    m_ko_hash_history.clear();
    m_ko_hash_history.push_back(board.get_ko_hash());
}

bool FastState::is_move_legal(int color, int vertex) {
    return vertex == FastBoard::PASS ||
           vertex == FastBoard::RESIGN ||
           (vertex != m_komove &&
                board.get_square(vertex) == FastBoard::EMPTY &&
                !board.is_suicide(vertex, color));
}

void FastState::play_move(int vertex) {
    play_move(board.m_tomove, vertex);
}

void FastState::play_move(int color, int vertex) {
    board.m_hash ^= Zobrist::zobrist_ko[m_komove];
    if (vertex == FastBoard::PASS) {
        // No Ko move
        m_komove = 0;
    } else {
        m_komove = board.update_board(color, vertex);
    }
    board.m_hash ^= Zobrist::zobrist_ko[m_komove];

    m_lastmove = vertex;
    m_movenum++;

    if (board.m_tomove == color) {
        board.m_hash ^= Zobrist::zobrist_blacktomove;
    }
    board.m_tomove = !color;

    board.m_hash ^= Zobrist::zobrist_pass[get_passes()];
    if (vertex == FastBoard::PASS) {
        increment_passes();
    } else {
        set_passes(0);
    }
    board.m_hash ^= Zobrist::zobrist_pass[get_passes()];

    m_ko_hash_history.push_back(board.get_ko_hash());
}

size_t FastState::get_movenum() const {
    return m_movenum;
}

int FastState::get_last_move(void) const {
    return m_lastmove;
}

int FastState::get_passes() const {
    return m_passes;
}

void FastState::set_passes(int val) {
    m_passes = val;
}

void FastState::increment_passes() {
    m_passes++;
    if (m_passes > 4) m_passes = 4;
}

int FastState::get_to_move() const {
    return board.m_tomove;
}

void FastState::set_to_move(int tom) {
    board.set_to_move(tom);
}

void FastState::display_state() {
    myprintf("\nPasses: %d            Black (X) Prisoners: %d\n",
             m_passes, board.get_prisoners(FastBoard::BLACK));
    if (board.black_to_move()) {
        myprintf("Black (X) to move");
    } else {
        myprintf("White (O) to move");
    }
    myprintf("    White (O) Prisoners: %d\n",
             board.get_prisoners(FastBoard::WHITE));

    board.display_board(get_last_move());
}

float FastState::final_score() const {
    return board.area_score(get_komi());
}

float FastState::get_komi() const {
    return m_komi;
}

bool FastState::superko(void) const {
    auto first = crbegin(m_ko_hash_history);
    auto last = crend(m_ko_hash_history);

    auto res = std::find(++first, last, board.get_ko_hash());

    return (res != last);
}

bool FastState::superko_move(const int color, const int i) const {
    auto newhash = board.test_update_ko_hash(color, i);
    auto first = cbegin(m_ko_hash_history);
    auto last = cend(m_ko_hash_history);

    auto res = std::find(first, last, newhash);
    return (res != last);
}
