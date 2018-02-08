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
#include "UCTSearch.h"

#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <stack>

#include "FastBoard.h"
#include "GTP.h"
#include "GameState.h"
#include "ThreadPool.h"
#include "Timing.h"
#include "Training.h"
#include "Utils.h"

using namespace Utils;

UCTSearch::UCTSearch() {
    set_playout_limit(cfg_max_playouts);
    m_root = std::make_unique<UCTNode>(FastBoard::PASS, 0.0f, 0.5f);
}

void UCTSearch::set_gamestate(const GameState & g) {

    m_rootstate = g;
    m_root = std::make_unique<UCTNode>(FastBoard::PASS, 0.0f, 0.5f);
    m_nodes = m_root->count_nodes();
}


bool UCTSearch::play_simulation(const GameState & state, UCTNode* const node) {
    
    auto currstate = state;
        
    bool result_valid = false;
    float result_eval;
    
    std::stack<UCTNode*> stacks;
    stacks.push(node);
    
    while (true) {
    
        auto node = stacks.top();
    
        node->virtual_loss();
    
        if (!node->has_children()) {
                if (currstate.get_passes() >= 2) {
                    result_eval = currstate.eval();
                    result_valid = true;
    
                } else if (m_nodes < MAX_TREE_SIZE) {
                    float eval;
                    auto success = node->create_children(m_nodes, currstate, eval);
                    if (success) {
                        result_eval = eval;
                        result_valid = true;
                    }
                } else {
                    result_eval = node->eval_state(currstate);
                    result_valid = true;
                }
        }
    
        if (result_valid)
            break;
    
        if (node->has_children()) {
            const auto color = currstate.get_to_move();
            auto next = node->uct_select_child(color);
    
            auto move = next->get_move();
    
            if (move != FastBoard::PASS) {
                currstate.play_move(move);
                stacks.push(next);
            } else {
                currstate.play_pass();
                stacks.push(next);
            }
        } else {
            // invalid path
            break;
        }
    }
    
    while (!stacks.empty()) {
    
        auto node = stacks.top();
        stacks.pop();
    
        if (result_valid) {
            node->update(result_eval);
        }
    
        node->virtual_loss_undo();
    }
    
    return result_valid;
}


void UCTSearch::dump_stats(FastState & state, UCTNode & parent) {
    if (cfg_quiet || !parent.has_children()) {
        return;
    }

    const int color = state.get_to_move();

    // sort children, put best move on top
    parent.sort_children(color);


    if (parent.get_first_child()->first_visit()) {
        return;
    }

    int movecount = 0;
    for (const auto& node : parent.get_children()) {
        // Always display at least two moves. In the case there is
        // only one move searched the user could get an idea why.
        if (++movecount > 2 && !node->get_visits()) break;

        std::string tmp = state.move_to_text(node->get_move());
        std::string pvstring(tmp);

        myprintf("%4s -> %7d (V: %5.2f%%) (N: %5.2f%%) PV: ",
            tmp.c_str(),
            node->get_visits(),
            node->get_eval(color)*100.0f,
            node->get_score() * 100.0f);

        FastState tmpstate = state;

        tmpstate.play_move(node->get_move());
        pvstring += " " + get_pv(tmpstate, *node);

        myprintf("%s\n", pvstring.c_str());
    }
}

bool UCTSearch::should_resign(float bestscore) {

    if (cfg_resignpct == 0) {
        // resign not allowed
        return false;
    }

    const auto visits = m_root->get_visits();
    if (visits < std::min(500, cfg_max_playouts))  {
        // low visits
        return false;
    }

    constexpr auto move_threshold = FastBoard::BOARDSQ / 4;
    const auto movenum = m_rootstate.get_movenum();
    if (movenum <= move_threshold) {
        // too early in game to resign
        return false;
    }

    const auto is_default_cfg_resign = cfg_resignpct < 0;
    const auto resign_threshold =
        0.01f * (is_default_cfg_resign ? 10 : cfg_resignpct);
    if (bestscore > resign_threshold) {
        // eval > cfg_resign
        return false;
    }

    return true;
}

int UCTSearch::get_best_move() {
    int color = m_rootstate.get_to_move();

    // Make sure best is first
    m_root->sort_children(color);

    // Check whether to randomize the best move proportional
    // to the playout counts, early game only.
    auto movenum = int(m_rootstate.get_movenum());
    if (movenum < cfg_random_cnt) {
        m_root->randomize_first_proportionally();
    }

    int bestmove = m_root->get_first_child()->get_move();

    // do we have statistics on the moves?
    if (m_root->get_first_child() != nullptr) {
        if (m_root->get_first_child()->first_visit()) {
            return bestmove;
        }
    }

    float bestscore = m_root->get_first_child()->get_eval(color);

    // if we aren't passing, should we consider resigning?
    if (bestmove != FastBoard::PASS) {
        if (should_resign(bestscore)) {
            myprintf("Eval (%.2f%%) looks bad. Resigning.\n",
                     100.0f * bestscore);
            bestmove = FastBoard::RESIGN;
        }
    }

    return bestmove;
}

std::string UCTSearch::get_pv(FastState & state, UCTNode& parent) {
    if (!parent.has_children()) {
        return std::string();
    }

    auto& best_child = parent.get_best_root_child(state.get_to_move());
    if (best_child.first_visit()) {
        return std::string();
    }
    auto best_move = best_child.get_move();
    auto res = state.move_to_text(best_move);

    state.play_move(best_move);

    auto next = get_pv(state, best_child);
    if (!next.empty()) {
        res.append(" ").append(next);
    }
    return res;
}

void UCTSearch::dump_analysis(int playouts) {
    if (cfg_quiet) {
        return;
    }

    GameState tempstate = m_rootstate;
    int color = tempstate.get_to_move();

    std::string pvstring = get_pv(tempstate, *m_root);
    float winrate = 100.0f * m_root->get_eval(color);
    myprintf("Playouts: %d, Win: %5.2f%%, PV: %s\n",
             playouts, winrate, pvstring.c_str());
}

int UCTSearch::think(int color) {

    m_nodes = m_root->count_nodes();
    
    // set side to move
    m_rootstate.set_to_move(color);

    // set up timing info
    Time start;

    myprintf("Thinking ...\n");

    // create a sorted list off legal moves (make sure we
    // play something legal and decent even in time trouble)
    float root_eval;
    if (!m_root->has_children()) {
        m_root->create_children(m_nodes, m_rootstate, root_eval);
    } else {
        root_eval = m_root->get_eval(color);
    }
    if (cfg_noise) {
        m_root->dirichlet_noise(0.25f, 0.03f);
    }

    myprintf("NN eval=%f\n",
             (color == FastBoard::BLACK ? root_eval : 1.0f - root_eval));

    std::atomic<int> playouts{0};
    std::atomic<bool> running{true};

    int cpus = cfg_num_threads;
    ThreadGroup tg(thread_pool);
    for (int i = 1; i < cpus; i++) {
        tg.add_task([&]() {
            do {
                if (play_simulation(m_rootstate, m_root.get()))
                    playouts++;
            } while(running && playouts < m_maxplayouts);
        });
    }

    int last_update = 0;
    do {
        if (play_simulation(m_rootstate, m_root.get()))
            playouts++;

        Time elapsed;
        int elapsed_centis = Time::timediff_centis(start, elapsed);

        // output some stats every few seconds
        // check if we should still search
        if (elapsed_centis - last_update > 250) {
            last_update = elapsed_centis;
            dump_analysis(static_cast<int>(playouts));
        }

    } while(playouts < m_maxplayouts);

    // stop the search
    running = false;
    tg.wait_all();

    // display search info
    myprintf("\n");

    dump_stats(m_rootstate, *m_root);
    record(m_rootstate, *m_root);

    Time elapsed;
    int elapsed_centis = Time::timediff_centis(start, elapsed);
    if (elapsed_centis+1 > 0) {
        myprintf("%d visits, %d nodes, %d playouts, %d n/s\n\n",
                 m_root->get_visits(),
                 static_cast<int>(m_nodes),
                 static_cast<int>(playouts),
                 (playouts * 100) / (elapsed_centis+1));
    }
    int bestmove = get_best_move();
    m_rootstate.play_move(bestmove);
    m_root = m_root->find_new_root(bestmove);
    return bestmove;
}

void UCTSearch::set_playout_limit(int playouts) {
    static_assert(std::is_convertible<decltype(playouts),
                                      decltype(m_maxplayouts)>::value,
                  "Inconsistent types for playout amount.");
    if (playouts == 0) {
        m_maxplayouts = std::numeric_limits<decltype(m_maxplayouts)>::max();
    } else {
        m_maxplayouts = playouts;
    }
}

void UCTSearch::record(GameState& state, UCTNode& root) {
    auto step = TimeStep{};
    step.to_move = state.get_to_move();
    step.planes = Network::NNPlanes{};
    Network::gather_features(&state, step.planes);

    auto result =
        Network::get_scored_moves(&state, Network::Ensemble::DIRECT, 0);
    step.net_winrate = result.second;

    const auto& best_node = root.get_best_root_child(step.to_move);
    step.root_uct_winrate = root.get_eval(step.to_move);
    step.child_uct_winrate = best_node.get_eval(step.to_move);
    step.bestmove_visits = best_node.get_visits();

    step.probabilities.resize((19 * 19) + 1);

    // Get total visit amount. We count rather
    // than trust the root to avoid ttable issues.
    auto sum_visits = 0.0;
    for (const auto& child : root.get_children()) {
        sum_visits += child->get_visits();
    }

    // In a terminal position (with 2 passes), we can have children, but we
    // will not able to accumulate search results on them because every attempt
    // to evaluate will bail immediately. So in this case there will be 0 total
    // visits, and we should not construct the (non-existent) probabilities.
    if (sum_visits <= 0.0) {
        return;
    }

    for (const auto& child : root.get_children()) {
        auto prob = static_cast<float>(child->get_visits() / sum_visits);
        auto move = child->get_move();
        if (move != FastBoard::PASS) {
            step.probabilities[move] = prob;
        } else {
            step.probabilities[FastBoard::BOARDSQ] = prob;
        }
    }

    Training::m_data.emplace_back(step);
}


std::vector<TimeStep> Training::m_data{};


void Training::clear_training() {
    Training::m_data.clear();
}

