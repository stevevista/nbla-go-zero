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

using namespace Utils;

void FastState::init_game(int size, float komi) {
    board.reset_board(size);

    m_tomove = FastBoard::BLACK;
    m_movenum = 0;

    m_komove = -1;
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
    reset_board();

    m_tomove = FastBoard::BLACK;
    m_movenum = 0;
    m_passes = 0;
    m_komove = -1;
    m_lastmove = 0;

    m_ko_hash_history.clear();
    m_ko_hash_history.push_back(board.get_ko_hash());
}

void FastState::reset_board(void) {
    board.reset_board(19);
    m_tomove = FastBoard::BLACK;
}

bool FastState::is_move_legal(int color, int vertex) const {
    
    if (vertex == FastBoard::PASS ||
           vertex == FastBoard::RESIGN)
        return true;

    if (vertex == m_komove)
        return false;

    std::uint64_t ko_hash;
    if (board.fast_test_move(color, vertex, ko_hash)) {
        auto res = std::find(m_ko_hash_history.begin(), m_ko_hash_history.end(), ko_hash);
        // not superko
        return (res == m_ko_hash_history.end());
    }

    return false;
}


void FastState::play_move(int vertex) {
    play_move(m_tomove, vertex);
}

void FastState::play_move(int color, int vertex) {

    if (vertex != FastBoard::PASS) {

        m_komove = board.update_board(color, vertex);
        m_lastmove = vertex;
        m_movenum++;
        m_tomove = !color;

        if (get_passes() > 0) {
            set_passes(0);
        }
    } else {

        m_movenum++;

        m_lastmove = FastBoard::PASS;

        m_komove = -1;
        m_tomove = !color;

        increment_passes();
    }
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
    return m_tomove;
}

bool FastState::black_to_move() const {
    return m_tomove == FastBoard::BLACK;
}

bool FastState::white_to_move() const {
    return m_tomove == FastBoard::WHITE;
}

void FastState::set_to_move(int tomove) {
    m_tomove = tomove;
}

void FastState::display_state() {
    myprintf("\nPasses: %d\n",
             m_passes);
    if (black_to_move()) {
        myprintf("Black (X) to move");
    } else {
        myprintf("White (O) to move");
    }

    board.display_board(get_last_move());
}

std::string FastState::move_to_text(int move) {
    return board.move_to_text(move);
}

float FastState::final_score() const {
    return board.area_score(get_komi());
}

float FastState::get_komi() const {
    return m_komi;
}

void FastState::play_pass(void) {
    play_move(get_to_move(), FastBoard::PASS);
}


float FastState::eval() const {
    return board.playout(get_to_move(), m_komove, 100, get_komi());
}
