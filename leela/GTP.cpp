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
#include "GTP.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <cstdarg>
#include <iostream>
#include <sstream>

#include "FastBoard.h"
#include "GameState.h"
#include "Network.h"
#include "SMP.h"
#include "UCTSearch.h"
#include "Utils.h"
#include "NNCache.h"
#include "Zobrist.h"

using namespace Utils;

// Configuration flags
int cfg_num_threads;
int cfg_max_playouts = 1600;
int cfg_max_visits;
int cfg_resignpct;
int cfg_random_cnt;
std::uint64_t cfg_rng_seed;
float cfg_puct;
float cfg_softmax_temp;
float cfg_fpu_reduction;
std::string cfg_weightsfile;
std::string cfg_logfile;
FILE* cfg_logfile_handle;
bool cfg_quiet;
std::string cfg_options_str;
int cfg_noise;

void GTP::setup_default_parameters() {

    cfg_num_threads = std::max(1, std::min(SMP::get_num_cpus(), MAX_CPUS));
    cfg_max_visits = std::numeric_limits<decltype(cfg_max_visits)>::max();
    cfg_puct = 0.8f;
    cfg_softmax_temp = 1.0f;
    cfg_fpu_reduction = 0.25f;
    // see UCTSearch::should_resign
    cfg_resignpct = -1;
    cfg_noise = true;
    cfg_random_cnt = 0;
    cfg_logfile_handle = nullptr;
    cfg_quiet = false;

    // C++11 doesn't guarantee *anything* about how random this is,
    // and in MinGW it isn't random at all. But we can mix it in, which
    // helps when it *is* high quality (Linux, MSVC).
    std::random_device rd;
    std::ranlux48 gen(rd());
    std::uint64_t seed1 = (gen() << 16) ^ gen();
    // If the above fails, this is one of our best, portable, bets.
    std::uint64_t seed2 = std::chrono::high_resolution_clock::
        now().time_since_epoch().count();
    cfg_rng_seed = seed1 ^ seed2;
}

// Setup global objects after command line has been parsed
void init_global_objects() {
    
    thread_pool.initialize(cfg_num_threads);
    
        // Use deterministic random numbers for hashing
    auto rng = std::make_unique<Random>(5489);
    FastBoard::init_board();
    Zobrist::init_zobrist(*rng);
    
        // Initialize the main thread RNG.
        // Doing this here avoids mixing in the thread_id, which
        // improves reproducibility across platforms.
    Random::get_Rng().seedrandom(cfg_rng_seed);
    
    NNCache::get_NNCache().set_size_from_playouts(cfg_max_playouts);
    
    // Initialize network
    Network::initialize();
}

void parse_leela_commandline(int argc, char *argv[]) {
    
    bool seed_set = false;
    for (int i = 1; i < argc; i++) {
    
            auto opt = std::string(argv[i]); 
    
            if (opt == "--threads" || opt == "-t") {
                int num_threads = std::stoi(argv[++i]);
                if (num_threads > cfg_num_threads) {
                    myprintf("Clamping threads to maximum = %d\n", cfg_num_threads);
                } else if (num_threads != cfg_num_threads) {
                    myprintf("Using %d thread(s).\n", num_threads);
                    cfg_num_threads = num_threads;
                }
            } 
        else if (opt == "--playouts" || opt == "-p") {
            cfg_max_playouts = std::stoi(argv[++i]);
        }
            else if (opt == "--visits" || opt == "-v") {
                cfg_max_visits = std::stoi(argv[++i]);
            }
            else if (opt == "--resignpct" || opt == "-r") {
                cfg_resignpct = std::stoi(argv[++i]);
            }
            else if (opt == "--randomcnt" || opt == "-m") {
                cfg_random_cnt = std::stoi(argv[++i]);
            }
            else if (opt == "--seed" || opt == "-s") {
                seed_set = true;
                cfg_rng_seed = std::stoull(argv[++i]);
                if (cfg_num_threads > 1) {
                    myprintf("Seed specified but multiple threads enabled.\n");
                    myprintf("Games will likely not be reproducible.\n");
                }
            }
            else if (opt == "--weights" || opt == "-w") {
                cfg_weightsfile = argv[++i];
            }
            else if (opt == "--logfile" || opt == "-l") {
                cfg_logfile = argv[++i];
                myprintf("Logging to %s.\n", cfg_logfile.c_str());
                cfg_logfile_handle = fopen(cfg_logfile.c_str(), "a");
            }
            else if (opt == "--quiet" || opt == "-q") {
                cfg_quiet = true;
            }
            else if (opt == "--puct") {
                cfg_puct = std::stof(argv[++i]);
            }
            else if (opt == "--softmax_temp") {
                cfg_softmax_temp = std::stof(argv[++i]);
            }
        else if (opt == "--fpu_reduction") {
            cfg_fpu_reduction = std::stof(argv[++i]);
        }
    }
    
    if (cfg_weightsfile.empty()) {
        myprintf("A network weights file is required to use the program.\n");
        throw std::runtime_error("A network weights file is required to use the program");
    }
    
    myprintf("RNG seed: %llu\n", cfg_rng_seed);
    
    auto out = std::stringstream{};
    for (auto i = 1; i < argc; i++) {
        out << " " << argv[i];
    }
    if (!seed_set) {
        out << " --seed " << cfg_rng_seed;
    }
    cfg_options_str = out.str();
}
