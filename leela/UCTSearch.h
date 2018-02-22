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

#include "FastBoard.h"
#include "GameState.h"
#include "UCTNode.h"
#include "Network.h"

struct TimeStep {
    Network::NNPlanes features;
    std::vector<float> probabilities;
    int to_move;
};

class UCTSearch {
public:
    /*
        Maximum size of the tree in memory. Nodes are about
        48 bytes, so limit to ~1.2G on 32-bits and about 5.5G
        on 64-bits.
    */
    static constexpr auto MAX_TREE_SIZE =
        (sizeof(void*) == 4 ? 25'000'000 : 100'000'000);

    UCTSearch(GameState& g);
    int think(int color, std::vector<TimeStep>& steps);
    void set_playout_limit(int playouts);
    void set_visit_limit(int visits);
    
private:
    bool play_simulation(const GameState& currstate, UCTNode* const node);
    bool stop_thinking() const;
    void increment_playouts();
    void dump_stats(const FastState& state, UCTNode& parent);
    std::string get_pv(FastState& state, UCTNode& parent);
    void dump_analysis(int playouts);
    bool should_resign(float bestscore);

    GameState & m_rootstate;
    std::unique_ptr<UCTNode> m_root;
    std::atomic<int> m_nodes{0};
    std::atomic<int> m_playouts{0};
    
    int m_maxplayouts;
    int m_maxvisits;
};



#endif
