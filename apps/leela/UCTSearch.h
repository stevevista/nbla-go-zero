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

#ifndef UCTSEARCH_H_INCLUDED
#define UCTSEARCH_H_INCLUDED

#include <atomic>
#include <memory>
#include <string>
#include <tuple>

#include "GameState.h"
#include "UCTNode.h"


class UCTSearch {
public:
    /*
        Maximum size of the tree in memory. Nodes are about
        48 bytes, so limit to ~1.2G on 32-bits and about 5.5G
        on 64-bits.
    */
    static constexpr auto MAX_TREE_SIZE = 300000000;

    UCTSearch();
    void set_gamestate(const GameState& g);
    int think(int color);
    void set_playout_limit(int playouts);


private:
    bool play_simulation(const GameState& currstate, UCTNode* const node);

    void record(GameState& state, UCTNode& node);
    void dump_stats(FastState& state, UCTNode& parent);
    std::string get_pv(FastState& state, UCTNode& parent);
    void dump_analysis(int playouts);
    bool should_resign(float bestscore);
    int get_best_move();

    GameState m_rootstate;
    std::unique_ptr<UCTNode> m_root;
    std::atomic<int> m_nodes{0};
    int m_maxplayouts;
};


#endif
