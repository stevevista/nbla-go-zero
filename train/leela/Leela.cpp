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

#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cctype>
#include <string>
#include <sstream>
#include <cmath>
#include <climits>
#include <algorithm>
#include <random>
#include <chrono>

#include "config.h"
#include "Utils.h"
#include "GameState.h"
#include "Leela.h"
#include "UCTNode.h"
#include "Network.h"
#include "NNCache.h"
#include "GTP.h"

using namespace Utils;


std::string cfg_weightsfile;
// Configuration flags
int cfg_num_threads;
int cfg_max_playouts;
int cfg_resignpct;
int cfg_noise;
int cfg_random_cnt;
std::uint64_t cfg_rng_seed;
float cfg_puct;
float cfg_softmax_temp;

std::string cfg_logfile;
FILE* cfg_logfile_handle;
bool cfg_quiet;


void setup_default_parameters() {

    cfg_num_threads = std::max(1, std::min(SMP::get_num_cpus(), MAX_CPUS));
    cfg_max_playouts = std::numeric_limits<decltype(cfg_max_playouts)>::max();
    cfg_puct = 0.85f;
    cfg_softmax_temp = 1.0f;
    // see UCTSearch::should_resign
    cfg_resignpct = -1;
    cfg_noise = false;
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
    Zobrist::init_zobrist(*rng);

    // Initialize the main thread RNG.
    // Doing this here avoids mixing in the thread_id, which
    // improves reproducibility across platforms.
    Random::get_Rng().seedrandom(cfg_rng_seed);

    NNCache::get_NNCache().set_size_from_playouts(cfg_max_playouts);

    // Initialize network
    Network::initialize();
}

bool Leela::load_weights(const std::string& path) {
    bool ret = Network::load_weights(path);
    if (ret)
        cfg_weightsfile = path;
    return ret;
}


Leela::Leela()
{
    
    static bool inited = false;
    if (!inited) {
    
        // Set up engine parameters
        setup_default_parameters();
        init_global_objects();
        inited = true;
    }
    
    search = std::make_unique<UCTSearch>();
    game = std::make_shared<GameState>();
    game->init_game(19, 7.5);
}


void Leela::clear_cache() {
    NNCache::get_NNCache().clear();
}
/*

{"cmd":"selfplay",
"options":{"playouts":"1600","resignation_percent":"1","noise":"true","randomcnt":"30"}}
*/
int Leela::selfplay(int playouts, const std::string& sgffile, std::function<void(int, int[])> callback) {

    // reset engine
    clear_board();

    cfg_resignpct = 1;
    cfg_random_cnt = 30;
    cfg_noise = true;
    search->set_playout_limit(playouts);

    int board[361];
    auto fill_board = [&]() {
        auto& info = game->get_past_board(0);
        for (int i=0; i<361; i++) {
            auto color = info[i];
            if (color == FastBoard::BLACK)
                board[i] = 1;
            else if (color == FastBoard::WHITE)
                board[i] = 2;
            else
                board[i] = 0;
        }
    };

    if (callback) {
        fill_board();
        callback(-1, board);
    }
    
    int move_cnt = 0;
    int winner = FastBoard::EMPTY;
    int prev_move = -1;
    std::vector<int> move_history;
    
    for(;;) {
        auto who = game->get_to_move();
        int move = search->think(who);
        game->play_move(who, move);
        move_history.push_back(move);
        move_cnt++;

        if (callback) {
            fill_board();
            callback(move, board);
        }
    
        //    game->display_state();

        std::string vertex = game->move_to_text(move);
        std::cerr << (who == FastBoard::BLACK ? "B" : "W") << " " << vertex << std::endl;
    
        if (move == FastBoard::RESIGN) {
            winner = !who;
            break;
        }
        if (move == FastBoard::PASS && prev_move == FastBoard::PASS) {
            break;
        }
    
        prev_move = move;
    
        if (move_cnt >= 361*2)
            break;
    }
    
    // Nobody resigned, we will have to count
    if (winner == FastBoard::EMPTY) {
        auto score = game->eval();
        if (score < 0.5) {
            winner = FastBoard::WHITE;
        } else if (score > 0.5) {
            winner = FastBoard::BLACK;
        }
    }
    
    int result = 0;
    if (winner == FastBoard::BLACK)
        result = 1;
    else if (winner == FastBoard::WHITE)
        result = -1;
    else
        result = 0;

    dump_sgf(sgffile, move_history);

    std::cerr << "final score: " << result << std::endl;

    return result;
}

    
bool Leela::dump_sgf(const std::string& path, const std::vector<int>& move_history) const {
    std::ofstream ofs(path);
    if (ofs.fail())
        return false;

    int compcolor = 0;

    float komi = game->get_komi();
    float score = game->final_score();

    std::string header;
    std::string moves;
    
    int size = FastBoard::BOARDSIZE;
    time_t now;
    time(&now);
    char timestr[sizeof "2017-10-16"];
    strftime(timestr, sizeof timestr, "%F", localtime(&now));
    
    header.append("(;GM[1]FF[4]RU[Chinese]");
    header.append("DT[" + std::string(timestr) + "]");
    header.append("SZ[" + std::to_string(size) + "]");
    header.append("KM[" + boost::format("%.1f", komi) + "]");
    
    auto leela_name = std::string{PROGRAM_NAME};
    leela_name.append(" " + std::string(PROGRAM_VERSION));
    if (!cfg_weightsfile.empty()) {
        leela_name.append(" " + cfg_weightsfile.substr(0, 8));
    }
    
    if (compcolor == FastBoard::WHITE) {
        header.append("PW[" + leela_name + "]");
        header.append("PB[Human]");
    } else {
        header.append("PB[" + leela_name + "]");
        header.append("PW[Human]");
    }
    
    moves.append("\n");
    
    int counter = 0;
    
    int color = FastBoard::BLACK;
    bool resigned = false;
    for (auto move : move_history) {
        if (move == FastBoard::RESIGN) {
            resigned = true;
            break;
        }
    
        std::string movestr = game->board.move_to_text_sgf(move);
        if (color == FastBoard::BLACK) {
            moves.append(";W[" + movestr + "]");
        } else {
            moves.append(";B[" + movestr + "]");
        }
        if (++counter % 10 == 0) {
            moves.append("\n");
        }
    
        color = !color;
    }
    
    if (!resigned) {
            if (score > 0.0f) {
                header.append("RE[B+" + boost::format("%.1f", score) + "]");
            } else {
                header.append("RE[W+" + boost::format("%.1f", -score) + "]");
            }
    } else {
            if (color == FastBoard::WHITE) {
                header.append("RE[B+Resign]");
            } else {
                header.append("RE[W+Resign]");
            }
    }
    
    header.append("\nC[" + std::string{PROGRAM_NAME} + " options:]");
    
    std::string sgf_text(header);
    sgf_text.append("\n");
    sgf_text.append(moves);
    sgf_text.append(")\n");

    ofs << sgf_text;
    return true;
}
    
void Leela::clear_board() {
    // Initialize the board.
    game->reset_game();
    search->set_gamestate(*game);
}
    
void Leela::komi(float v) {
    game->set_komi(v);
}
