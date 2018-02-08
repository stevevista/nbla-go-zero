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

#ifndef TRAINING_H_INCLUDED
#define TRAINING_H_INCLUDED

#include "config.h"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>
#include "Network.h"

class TimeStep {
public:
    Network::NNPlanes planes;
    std::vector<float> probabilities;
    int to_move;
    float net_winrate;
    float root_uct_winrate;
    float child_uct_winrate;
    int bestmove_visits;
};

class Training {
public:
    static void clear_training();

    static std::vector<TimeStep> m_data;
};

#endif
