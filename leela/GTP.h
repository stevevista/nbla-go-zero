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

#ifndef GTP_H_INCLUDED
#define GTP_H_INCLUDED

#include "config.h"

#include <cstdio>
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <queue>
#include <mutex>  
#include <condition_variable> 

#include "GameState.h"
#include "UCTSearch.h"


extern int cfg_num_threads;
extern int cfg_max_playouts;
extern int cfg_max_visits;
extern int cfg_resignpct;
extern int cfg_random_cnt;
extern std::uint64_t cfg_rng_seed;
extern float cfg_puct;
extern float cfg_softmax_temp;
extern float cfg_fpu_reduction;
extern std::string cfg_logfile;
extern std::string cfg_weightsfile;
extern FILE* cfg_logfile_handle;
extern bool cfg_quiet;
extern std::string cfg_options_str;
extern int cfg_noise;


void init_global_objects();
void parse_commandline(int argc, char *argv[]);

class GTP {
public:
    static void setup_default_parameters();
};



#endif
