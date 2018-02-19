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

#include <cstdint>
#include <algorithm>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>

#include "GTP.h"
#include "GameState.h"
#include "Network.h"
#include "NNCache.h"
#include "Random.h"
#include "ThreadPool.h"
#include "Utils.h"
#include "Zobrist.h"

using namespace Utils;

static void license_blurb() {
    printf(
        "Leela Zero  Copyright (C) 2017-2018  Gian-Carlo Pascutto and contributors\n"
        "This program comes with ABSOLUTELY NO WARRANTY.\n"
        "This is free software, and you are welcome to redistribute it\n"
        "under certain conditions; see the COPYING file for details.\n\n"
    );
}

static void parse_commandline(int argc, char *argv[]) {

    bool seed_set = false;
    for (int i = 1; i < argc; i++) {

        auto opt = std::string(argv[i]); 

        if (opt == "--help" or opt == "-h") {
            license_blurb();
            exit(0);
        } 
        else if (opt == "--gtp" or opt == "-g") {
            cfg_gtp_mode = true;
        } 
        else if (opt == "--threads" or opt == "-t") {
            int num_threads = std::stoi(argv[++i]);
            if (num_threads > cfg_num_threads) {
                myprintf("Clamping threads to maximum = %d\n", cfg_num_threads);
            } else if (num_threads != cfg_num_threads) {
                myprintf("Using %d thread(s).\n", num_threads);
                cfg_num_threads = num_threads;
            }
        } 
        else if (opt == "--playouts" or opt == "-p") {
            cfg_max_playouts = std::stoi(argv[++i]);
        }
        else if (opt == "--noponder") {
            cfg_allow_pondering = false;
        }
        else if (opt == "--visits" or opt == "-v") {
            cfg_max_visits = std::stoi(argv[++i]);
        }
        else if (opt == "--lagbuffer" or opt == "-b") {
            int lagbuffer = std::stoi(argv[++i]);
            if (lagbuffer != cfg_lagbuffer_cs) {
                myprintf("Using per-move time margin of %.2fs.\n", lagbuffer/100.0f);
                cfg_lagbuffer_cs = lagbuffer;
            }
        }
        else if (opt == "--resignpct" or opt == "-r") {
            cfg_resignpct = std::stoi(argv[++i]);
        }
        else if (opt == "--randomcnt" or opt == "-m") {
            cfg_random_cnt = std::stoi(argv[++i]);
        }
        else if (opt == "--noise" or opt == "-n") {
            cfg_noise = true;
        }
        else if (opt == "--seed" or opt == "-s") {
            seed_set = true;
            cfg_rng_seed = std::stoull(argv[++i]);
            if (cfg_num_threads > 1) {
                myprintf("Seed specified but multiple threads enabled.\n");
                myprintf("Games will likely not be reproducible.\n");
            }
        }
        else if (opt == "--dumbpass" or opt == "-d") {
            cfg_dumbpass = true;
        }
        else if (opt == "--weights" or opt == "-w") {
            cfg_weightsfile = argv[++i];
        }
        else if (opt == "--logfile" or opt == "-l") {
            cfg_logfile = argv[++i];
            myprintf("Logging to %s.\n", cfg_logfile.c_str());
            cfg_logfile_handle = fopen(cfg_logfile.c_str(), "a");
        }
        else if (opt == "--quiet" or opt == "-q") {
            cfg_quiet = true;
        }
        else if (opt == "--gpu") {
            cfg_gpus = {std::stoi(argv[++i])};
        }
        else if (opt == "--full-tuner") {
            cfg_sgemm_exhaustive = true;
        }
        else if (opt == "--tune-only") {
            cfg_tune_only = true;
        }
        else if (opt == "--puct") {
            cfg_puct = std::stof(argv[++i]);
        }
        else if (opt == "--softmax_temp") {
            cfg_softmax_temp = std::stof(argv[++i]);
        }
    }

    if (cfg_max_visits < std::numeric_limits<decltype(cfg_max_visits)>::max() && cfg_allow_pondering) {
        myprintf("Nonsensical options: Playouts are restricted but "
                        "thinking on the opponent's time is still allowed. "
                        "Add --noponder if you want a weakened engine.\n");
        exit(EXIT_FAILURE);
    }

    if (cfg_weightsfile.empty()) {
        myprintf("A network weights file is required to use the program.\n");
        exit(EXIT_FAILURE);
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

// Setup global objects after command line has been parsed
void init_global_objects() {
    thread_pool.initialize(cfg_num_threads);

    // Use deterministic random numbers for hashing
    auto rng = std::make_unique<Random>(5489);
    Zobrist::init_zobrist(*rng);

    // Initialize the main thread RNG.
    // Doing this here avoids mixing in the thread_id, which
    // improves reproducibility across platforms.
    Random::get_Rng().seedrandom(cfg_rng_seed);

    NNCache::get_NNCache().set_size_from_playouts(cfg_max_playouts);

    // Initialize network
    Network::initialize();
}

int main (int argc, char *argv[]) {
    auto input = std::string{};

    // Set up engine parameters
    GTP::setup_default_parameters();
    parse_commandline(argc, argv);

    // Disable IO buffering as much as possible
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
    std::cin.setf(std::ios::unitbuf);

    setbuf(stdout, nullptr);
    setbuf(stderr, nullptr);
#ifndef WIN32
    setbuf(stdin, nullptr);
#endif

    if (!cfg_gtp_mode) {
        license_blurb();
    }

    init_global_objects();

    auto maingame = std::make_unique<GameState>();

    /* set board limits */
    auto komi = 7.5f;
    maingame->init_game(19, komi);

    for(;;) {
        if (!cfg_gtp_mode) {
            maingame->display_state();
            std::cout << "Leela: ";
        }

        if (std::getline(std::cin, input)) {
            Utils::log_input(input);
            GTP::execute(*maingame, input);
        } else {
            // eof or other error
            break;
        }

        // Force a flush of the logfile
        if (cfg_logfile_handle) {
            fclose(cfg_logfile_handle);
            cfg_logfile_handle = fopen(cfg_logfile.c_str(), "a");
        }
    }

    return 0;
}
