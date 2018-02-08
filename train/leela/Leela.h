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

#ifndef LEELA_AGENT_H_INCLUDED
#define LEELA_AGENT_H_INCLUDED

#include "config.h"
#include <string>
#include <vector>
#include <functional>
#include "GameState.h"
#include "UCTSearch.h"
#include "defs.h"

class NENG_API Leela {
    
protected:
    std::shared_ptr<GameState> game;
    std::unique_ptr<UCTSearch> search;
        
public:
    Leela();
    bool load_weights(const std::string& path);
    
    void clear_board();
    void komi(float);
    bool dump_sgf(const std::string& path, const std::vector<int>& move_history) const;

    void clear_cache();
    int selfplay(int playouts, const std::string& sgffile, std::function<void(int, int[])> callback);
};


#endif
