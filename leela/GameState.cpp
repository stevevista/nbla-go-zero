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

#include "GameState.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>

#include "FastBoard.h"
#include "FastState.h"
#include "UCTSearch.h"

void GameState::init_game(float komi) {
    FastState::init_game(komi);

    game_history.clear();
    game_history.emplace_back(std::make_shared<FastBoard>(board));

    m_resigned = FastBoard::EMPTY;
}

void GameState::reset_game() {
    FastState::reset_game();

    game_history.clear();
    game_history.emplace_back(std::make_shared<FastBoard>(board));

    m_resigned = FastBoard::EMPTY;
}

void GameState::play_move(int vertex) {
    play_move(get_to_move(), vertex);
}

void GameState::play_move(int color, int vertex) {
    if (vertex == FastBoard::RESIGN) {
        m_resigned = color;
    } else {
        FastState::play_move(color, vertex);
    }

    // cut off any leftover moves from navigating
    game_history.resize(m_movenum);
    game_history.emplace_back(std::make_shared<FastBoard>(board));
}

int GameState::who_resigned() const {
    return m_resigned;
}

bool GameState::has_resigned() const {
    return m_resigned != FastBoard::EMPTY;
}

void GameState::anchor_game_history(void) {
    // handicap moves don't count in game history
    m_movenum = 0;
    game_history.clear();
    game_history.emplace_back(std::make_shared<FastBoard>(board));
}

const FastBoard& GameState::get_past_board(int moves_ago) const {
    assert(moves_ago >= 0 && (unsigned)moves_ago <= m_movenum);
    assert(m_movenum + 1 <= game_history.size());
    return *game_history[m_movenum - moves_ago];
}
