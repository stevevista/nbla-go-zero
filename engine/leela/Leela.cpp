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
#include "TTable.h"
#include "Zobrist.h"
#include "Network.h"
#include "NNCache.h"
#include "GTP.h"
#include "SGFTree.h"

using namespace Utils;


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


void Leela::stop_ponder() {
    Utils::enable_ponder(false);
}

Leela::Leela(ILeelaModel* m, const std::string& logfile)
{
    
    static bool inited = false;
    if (!inited) {
    
        // Set up engine parameters
        GTP::setup_default_parameters();
        init_global_objects();
        inited = true;
    }

    if (!logfile.empty()) {
        cfg_logfile_handle = fopen(logfile.c_str(), "w");
    }
    
    Network::register_model(m);
    
    search = std::make_unique<UCTSearch>();
    game = std::make_shared<GameState>();
    game->init_game(19, 7.5);
}

int Leela::genmove() {
    Utils::enable_ponder(true);
	return genmove(game->get_to_move() == FastBoard::BLACK ? 1 : 0, true);
}

int Leela::genmove(int player, bool commit) {
    
    int who = player == 1 ? FastBoard::BLACK : FastBoard::WHITE;
    //if (who != game->get_to_move())
     //   game->play_pass();
    // start thinking
    Utils::enable_ponder(true);
    int move = search->think(game->get_to_move(), *game);
    if (commit)
        game->play_move(who, move);

    if (move == FastBoard::PASS)
        return -1;
    if (move == FastBoard::RESIGN)
        return -2;
    return move;
}

int Leela::get_color(int idx) const {
    auto x = idx % 19;
    auto y = idx / 19;
    auto vtx = game->board.get_vertex(x, y);
	auto color = game->board.get_square(vtx);
    if (color == FastBoard::BLACK)
        return 1;
    else if (color == FastBoard::WHITE)
        return 2;
    else
        return 0;
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
    cfg_dumbpass = true;
    cfg_noise = true;
    search->set_playout_limit(playouts);

    // Infinite thinking time set.
    game->set_timecontrol(0, 1 * 100, 0, 0);

    

    int board[361];
    auto fill_board = [&]() {
        for (int i=0; i<361; i++) {
            board[i] = get_color(i);
        }
    };

    if (callback) {
        fill_board();
        callback(-1, board);
    }
    
    int move_cnt = 0;
    int winner = FastBoard::EMPTY;
    int prev_move = -1;
    
    for(;;) {
        auto who = game->get_to_move();
        int move = search->think(who, *game, UCTSearch::NORMAL);
        game->play_move(who, move);
        move_cnt++;

        if (callback) {
            fill_board();
            int idx = -1;
            if (move != FastBoard::RESIGN && move != FastBoard::PASS) {
                auto xy = game->board.get_xy(move);
                idx = xy.first + xy.second*19;
            }
            callback(idx, board);
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
        float score = final_score();
        if (score < 0) {
            winner = FastBoard::WHITE;
        } else if (score > 0) {
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

    dump_sgf(sgffile);

    std::cerr << "final score: " << result << std::endl;

    return result;
}

    
bool Leela::dump_sgf(const std::string& path) const {
    std::ofstream ofs(path);
    if (ofs.fail())
        return false;

    auto sgf_text = SGFTree::state_to_string(*game, 0);
    ofs << sgf_text;
    return true;
}

void Leela::ponder_on_idle() {
    
    // now start pondering
    if (game->get_last_move() != FastBoard::RESIGN) {
        Utils::enable_ponder(true);
        search->ponder(*game);
    }
}
    
void Leela::pass(int player) {
    game->play_pass();
}
        
void Leela::resign(int player) {
    game->play_move(player == 1 ? FastBoard::BLACK : FastBoard::WHITE, FastBoard::RESIGN);
}
    
void Leela::play(int player, int move) {

    game->play_move(player == 1 ? FastBoard::BLACK : FastBoard::WHITE, move);
}
    
    void Leela::quit() {
    }
    
    void Leela::game_over() {
    }
    
    void Leela::set_timecontrol(int maintime, int byotime, int byostones, int byoperiods) {
    
        game->set_timecontrol(maintime * 100, byotime * 100, byostones, byoperiods);
            
    }
    
    
float Leela::final_score() {	
    float ftmp = game->final_score();
    /* white wins */
    if (ftmp < -0.1 || ftmp > 0.1) {
        return ftmp;
    } else {
        return 0;
    }
}


void Leela::ponder_enable() {
    
}
    
    std::string Leela::name() {
        return "Leela";
    }
    
    void Leela::time_left(int player, double t) {
    
        game->adjust_time(player == 1 ? FastBoard::BLACK : FastBoard::WHITE, t * 100, 0);
    }
    
    
void Leela::clear_board() {
    // Initialize the board.
    game->reset_game();
    std::make_unique<UCTSearch>().swap(search);
}
    
    void Leela::komi(float v) {
        game->set_komi(v);
    }

void Leela::heatmap(int rotation) const {

    auto vec = Network::get_scored_moves(
            game.get(), Network::Ensemble::DIRECT, rotation);
    Network::show_heatmap(game.get(), vec, false);

}


PolicyPlayer::PolicyPlayer(ILeelaModel* model, const std::string& logfile)
:Leela(model, logfile)
{}

std::string PolicyPlayer::name()
{
    return "policy";
}

int PolicyPlayer::genmove(int player, bool commit) {
    
    int who = player == 1 ? FastBoard::BLACK : FastBoard::WHITE;
    
    int move = FastBoard::PASS;

    auto result = Network::get_scored_moves(
                            game.get(), Network::Ensemble::DIRECT, 0);
    auto moves = result.first;
    std::stable_sort(moves.rbegin(), moves.rend());
    
    for (auto kv : moves) {
        if (game->is_move_legal(game->get_to_move(), kv.second)) {
                //game.display_state();
                //std::cout << "legal " <<kv.second << std::endl;
            move = kv.second;
            break;
        }
    }

    if (commit)
        game->play_move(who, move);

    if (move == FastBoard::PASS)
        return -1;
    if (move == FastBoard::RESIGN)
        return -2;
    return move;
}

void PolicyPlayer::ponder_on_idle() {

}
