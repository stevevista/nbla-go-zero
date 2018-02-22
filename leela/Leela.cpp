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
#include "Random.h"
#include "Zobrist.h"

using namespace Utils;


bool Leela::load_weights(const std::string& path) {

    size_t channels, residual_blocks;
    cfg_weightsfile = path;
    std::tie(channels, residual_blocks) = Network::load_network_file(cfg_weightsfile);
    return channels > 0;
}


Leela::Leela(const std::string& wpath)
{
    cfg_weightsfile = wpath;

    static bool inited = false;
    if (!inited) {
        // Set up engine parameters
        GTP::setup_default_parameters();
        init_global_objects();
        inited = true;
    }
    else {
        Network::load_network_file(cfg_weightsfile);
    }
    
    
    game = std::make_shared<GameState>();
    search = std::make_unique<UCTSearch>(*game);
    game->init_game(7.5);
}

/*

{"cmd":"selfplay",
"options":{"playouts":"1600","resignation_percent":"1","noise":"true","randomcnt":"30"}}
*/
int Leela::selfplay(int playouts, std::vector<TimeStep>& steps, const std::string& sgffile, std::function<void(int, int[])> callback) {

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
            auto color = info.get_square(i);
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
        int move = search->think(who, steps);
        game->play_move(who, move);
        move_history.push_back(move);
        move_cnt++;

        if (callback) {
            fill_board();
            callback(move, board);
        }
    
        //    game->display_state();

        std::string vertex = FastBoard::move_to_text(move);
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
        auto score = game->final_score();
        if (score < -0.1) {
            winner = FastBoard::WHITE;
        } else if (score > 0.1) {
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
    header.append("KM[" + format("%.1f", komi) + "]");
    
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
                header.append("RE[B+" + format("%.1f", score) + "]");
            } else {
                header.append("RE[W+" + format("%.1f", -score) + "]");
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
    search = std::make_unique<UCTSearch>(*game);
}
    
void Leela::komi(float v) {
    game->set_komi(v);
}
