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

#include "FastBoard.h"

#include <cassert>
#include <array>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <cctype>

#include "Utils.h"
#include "Zobrist.h"

using namespace Utils;

const int FastBoard::BIG;
const int FastBoard::PASS;
const int FastBoard::RESIGN;

std::array<std::vector<int>, FastBoard::BOARDSQ> FastBoard::NEIGHBORS;
std::array<std::vector<int>, FastBoard::BOARDSQ> FastBoard::DIAGS;

void FastBoard::init_board() {
    for (int y=0; y<BOARDSIZE; y++) {
        for (int x=0; x<BOARDSIZE; x++) {
            auto& n = NEIGHBORS[y*BOARDSIZE + x];
            auto& d = DIAGS[y*BOARDSIZE + x];

            if (y > 0) n.emplace_back((y-1)*BOARDSIZE + x);
            if (y < BOARDSIZE-1) n.emplace_back((y+1)*BOARDSIZE + x);
            if (x > 0) n.emplace_back(y*BOARDSIZE + x - 1);
            if (x < BOARDSIZE-1) n.emplace_back(y*BOARDSIZE + x + 1);

            if (y > 0 && x > 0) d.emplace_back((y-1)*BOARDSIZE + x -1);
            if (y > 0 && x < BOARDSIZE-1) d.emplace_back((y-1)*BOARDSIZE + x + 1);
            if (y < BOARDSIZE-1 && x > 0) d.emplace_back((y+1)*BOARDSIZE + x - 1);
            if (y < BOARDSIZE-1 && x < BOARDSIZE-1) d.emplace_back((y+1)*BOARDSIZE + x + 1);
        }
    }
}

FastBoard::square_t FastBoard::get_square(int vertex) const {

    assert(vertex >= 0 && vertex < BOARDSQ);
    return m_square[vertex];
}

FastBoard::square_t FastBoard::get_square(int x, int y) const {
    return m_square[y*BOARDSIZE + x];
}

void FastBoard::reset_board() {

    m_tomove = BLACK;
    m_prisoners[BLACK] = 0;
    m_prisoners[WHITE] = 0;

    std::fill(m_square.begin(), m_square.end(), EMPTY);

    calc_hash();
    calc_ko_hash();
}

bool FastBoard::is_suicide(int i, int color) const {

    for (auto ai : NEIGHBORS[i]) {
        if (m_square[ai] == EMPTY) {
            return false;
        } else if (m_square[ai] == color) {
            auto libs = m_libs[m_parent[ai]];
            if (libs > 1) {
                // connecting to live group = not suicide
                return false;
            }
        } else if (m_square[ai] == !color) {
            auto libs = m_libs[m_parent[ai]];
            if (libs <= 1) {
                // killing neighbour = not suicide
                return false;
            }
        }
    }

    // We played in a hole, friendlies had one liberty at most and
    // we did not kill anything. So we killed ourselves.
    return true;
}

int FastBoard::calc_reach_color(int color) const {
    auto reachable = 0;
    auto bd = std::vector<bool>(BOARDSQ, false);
    auto open = std::queue<int>();
    for (auto i = 0; i < BOARDSQ; i++) {
        if (m_square[i] == color) {
            reachable++;
            bd[i] = true;
            open.push(i);
        }
    }
    while (!open.empty()) {
        /* colored field, spread */
        auto vertex = open.front();
        open.pop();

        for (auto neighbor : NEIGHBORS[vertex]) {

            if (!bd[neighbor] && m_square[neighbor] == EMPTY) {
                reachable++;
                bd[neighbor] = true;
                open.push(neighbor);
            }
        }
    }
    return reachable;
}

// Needed for scoring passed out games not in MC playouts
float FastBoard::area_score(float komi) const {
    auto white = calc_reach_color(WHITE);
    auto black = calc_reach_color(BLACK);
    return black - white - komi;
}

void FastBoard::display_board(int lastmove) {

    myprintf("\n   ");
    print_columns();
    for (int j = BOARDSIZE-1; j >= 0; j--) {
        myprintf("%2d", j+1);
        if (lastmove == j*BOARDSIZE)
            myprintf("(");
        else
            myprintf(" ");
        for (int i = 0; i < BOARDSIZE; i++) {
            if (get_square(i,j) == WHITE) {
                myprintf("O");
            } else if (get_square(i,j) == BLACK)  {
                myprintf("X");
            } else if (starpoint(BOARDSIZE, i, j)) {
                myprintf("+");
            } else {
                myprintf(".");
            }
            if (lastmove == j*BOARDSIZE + i) myprintf(")");
            else if (i != BOARDSIZE-1 && lastmove == j*BOARDSIZE + i+1) myprintf("(");
            else myprintf(" ");
        }
        myprintf("%2d\n", j+1);
    }
    myprintf("   ");
    print_columns();
    myprintf("\n");
    myprintf("Hash: %llX Ko-Hash: %llX\n\n", get_hash(), get_ko_hash());
}

void FastBoard::print_columns() {
    for (int i = 0; i < BOARDSIZE; i++) {
        if (i < 25) {
            myprintf("%c ", (('a' + i < 'i') ? 'a' + i : 'a' + i + 1));
        }
        else {
            myprintf("%c ", (('A' + (i - 25) < 'I') ? 'A' + (i - 25) : 'A' + (i - 25) + 1));
        }
    }
    myprintf("\n");
}

void FastBoard::merge_strings(const int ip, const int aip) {

    /* merge stones */
    m_stones[ip] += m_stones[aip];

    /* loop over stones, update parents */
    int newpos = aip;

    do {
        // check if this stone has a liberty
        for (auto ai : NEIGHBORS[newpos]) {

            // for each liberty, check if it is not shared
            if (m_square[ai] == EMPTY) {
                // find liberty neighbors
                bool found = false;
                for (auto aai : NEIGHBORS[ai]) {

                    // friendly string shouldn't be ip
                    // ip can also be an aip that has been marked
                    if (m_square[aai] != EMPTY && m_parent[aai] == ip) {
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    m_libs[ip]++;
                }
            }
        }

        m_parent[newpos] = ip;
        newpos = m_next[newpos];
    } while (newpos != aip);

    /* merge stings */
    std::swap(m_next[aip], m_next[ip]);
}

bool FastBoard::is_eye(const int color, const int i) const {

    /* check for 4 neighbors of the same color */
    for (auto ai : NEIGHBORS[i]) {
        if (m_square[ai] != color)
            return false;
    }

    // 2 or more diagonals taken
    // 1 for side groups
    int colorcount[3];

    colorcount[BLACK] = 0;
    colorcount[WHITE] = 0;

    for (auto ai : DIAGS[i]) {
        colorcount[m_square[ai]]++;
    }

    if (DIAGS[i].size() == 4) {
        if (colorcount[!color] > 1) {
            return false;
        }
    } else {
        if (colorcount[!color]) {
            return false;
        }
    }

    return true;
}

std::string FastBoard::move_to_text(int move) {
    std::ostringstream result;

    int column = move % BOARDSIZE;
    int row = move / BOARDSIZE;

    assert(move == FastBoard::PASS || move == FastBoard::RESIGN || (row >= 0 && row < BOARDSIZE));
    assert(move == FastBoard::PASS || move == FastBoard::RESIGN || (column >= 0 && column < BOARDSIZE));

    if (move >= 0) {
        result << static_cast<char>(column < 8 ? 'A' + column : 'A' + column + 1);
        result << (row + 1);
    } else if (move == FastBoard::PASS) {
        result << "pass";
    } else if (move == FastBoard::RESIGN) {
        result << "resign";
    } else {
        result << "error";
    }

    return result.str();
}

std::string FastBoard::move_to_text_sgf(int move) const {
    std::ostringstream result;

    int column = move % BOARDSIZE;
    int row = move / BOARDSIZE;

    assert(move == FastBoard::PASS || move == FastBoard::RESIGN || (row >= 0 && row < BOARDSIZE));
    assert(move == FastBoard::PASS || move == FastBoard::RESIGN || (column >= 0 && column < BOARDSIZE));

    // SGF inverts rows
    row = BOARDSIZE - row - 1;

    if (move >= 0) {
        if (column <= 25) {
            result << static_cast<char>('a' + column);
        } else {
            result << static_cast<char>('A' + column - 26);
        }
        if (row <= 25) {
            result << static_cast<char>('a' + row);
        } else {
            result << static_cast<char>('A' + row - 26);
        }
    } else if (move == FastBoard::PASS) {
        result << "tt";
    } else if (move == FastBoard::RESIGN) {
        result << "tt";
    } else {
        result << "error";
    }

    return result.str();
}

bool FastBoard::starpoint(int size, int point) {
    int stars[3];
    int points[2];
    int hits = 0;

    if (size % 2 == 0 || size < 9) {
        return false;
    }

    stars[0] = size >= 13 ? 3 : 2;
    stars[1] = size / 2;
    stars[2] = size - 1 - stars[0];

    points[0] = point / size;
    points[1] = point % size;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (points[i] == stars[j]) {
                hits++;
            }
        }
    }

    return hits >= 2;
}

bool FastBoard::starpoint(int size, int x, int y) {
    return starpoint(size, y * size + x);
}

int FastBoard::get_prisoners(int side)  const {
    assert(side == WHITE || side == BLACK);

    return m_prisoners[side];
}

int FastBoard::get_to_move() const {
    return m_tomove;
}

bool FastBoard::black_to_move() const {
    return m_tomove == BLACK;
}

bool FastBoard::white_to_move() const {
    return m_tomove == WHITE;
}

void FastBoard::set_to_move(int tomove) {
    if (m_tomove != tomove) {
        m_hash ^= Zobrist::zobrist_blacktomove;
    }
    m_tomove = tomove;
}

std::string FastBoard::get_string(int vertex) const {
    std::string result;

    int start = m_parent[vertex];
    int newpos = start;

    do {
        result += move_to_text(newpos) + " ";
        newpos = m_next[newpos];
    } while (newpos != start);

    // eat last space
    assert(result.size() > 0);
    result.resize(result.size() - 1);

    return result;
}

std::string FastBoard::get_stone_list() const {
    std::string result;

    for (int i = 0; i < BOARDSQ; i++) {
        if (m_square[i] != EMPTY) {
            result += move_to_text(i) + " ";
        }
    }

    // eat final space, if any.
    if (result.size() > 0) {
        result.resize(result.size() - 1);
    }

    return result;
}


int FastBoard::remove_string(int i) {
    int pos = i;
    int removed = 0;
    int color = m_square[i];

    do {
        m_hash    ^= Zobrist::zobrist[m_square[pos]][pos];
        m_ko_hash ^= Zobrist::zobrist[m_square[pos]][pos];

        m_square[pos] = EMPTY;

        std::array<int, 4> nbr_pars;
        int nbr_par_cnt = 0;

        for (auto ai : NEIGHBORS[pos]) {

            if (m_square[ai] == EMPTY) continue;
    
            bool found = false;
            for (int n = 0; n < nbr_par_cnt; n++) {
                if (nbr_pars[n] == m_parent[ai]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                m_libs[m_parent[ai]]++;
                nbr_pars[nbr_par_cnt++] = m_parent[ai];
            }
        }

        m_hash    ^= Zobrist::zobrist[m_square[pos]][pos];
        m_ko_hash ^= Zobrist::zobrist[m_square[pos]][pos];

        removed++;
        pos = m_next[pos];
    } while (pos != i);

    return removed;
}

std::uint64_t FastBoard::calc_ko_hash(void) {
    auto res = Zobrist::zobrist_empty;

    for (int i = 0; i < BOARDSQ; i++) {
        res ^= Zobrist::zobrist[m_square[i]][i];
    }

    /* Tromp-Taylor has positional superko */
    m_ko_hash = res;
    return res;
}

std::uint64_t FastBoard::calc_hash(int komove) {
    auto res = Zobrist::zobrist_empty;

    for (int i = 0; i < BOARDSQ; i++) {
        res ^= Zobrist::zobrist[m_square[i]][i];
    }

    /* prisoner hashing is rule set dependent */
    res ^= Zobrist::zobrist_pris[0][m_prisoners[0]];
    res ^= Zobrist::zobrist_pris[1][m_prisoners[1]];

    if (m_tomove == BLACK) {
        res ^= Zobrist::zobrist_blacktomove;
    }

    res ^= Zobrist::zobrist_ko[komove];

    m_hash = res;

    return res;
}

std::uint64_t FastBoard::get_hash(void) const {
    return m_hash;
}

std::uint64_t FastBoard::get_ko_hash(void) const {
    return m_ko_hash;
}

int FastBoard::update_board(const int color, const int i) {
    assert(i != FastBoard::PASS);
    assert(m_square[i] == EMPTY);

    m_hash ^= Zobrist::zobrist[m_square[i]][i];
    m_ko_hash ^= Zobrist::zobrist[m_square[i]][i];

    m_square[i] = (square_t)color;
    m_next[i] = i;
    m_parent[i] = i;
    m_stones[i] = 1;

    m_hash ^= Zobrist::zobrist[m_square[i]][i];
    m_ko_hash ^= Zobrist::zobrist[m_square[i]][i];

    /* update neighbor liberties (they all lose 1) */
    int libs = 0;
    std::array<int, 4> nbr_pars;
    int nbr_par_cnt = 0;

    for (auto ai : NEIGHBORS[i]) {

        if (m_square[ai] == EMPTY) {
            libs++;
        } else {
            bool found = false;
            for (int n = 0; n < nbr_par_cnt; n++) {
                if (nbr_pars[n] == m_parent[ai]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                m_libs[m_parent[ai]]--;
                nbr_pars[nbr_par_cnt++] = m_parent[ai];
            }
        }
    }

    m_libs[i] = libs;

    auto captured_stones = 0;
    int captured_sq;

    for (auto ai : NEIGHBORS[i]) {

        if (m_square[ai] == !color) {
            if (m_libs[m_parent[ai]] <= 0) {
                int this_captured = remove_string(ai);
                captured_sq = ai;
                captured_stones += this_captured;
            }
        } else if (m_square[ai] == color) {
            int ip = m_parent[i];
            int aip = m_parent[ai];

            if (ip != aip) {
                if (m_stones[ip] >= m_stones[aip]) {
                    merge_strings(ip, aip);
                } else {
                    merge_strings(aip, ip);
                }
            }
        }
    }

    m_hash ^= Zobrist::zobrist_pris[color][m_prisoners[color]];
    m_prisoners[color] += captured_stones;
    m_hash ^= Zobrist::zobrist_pris[color][m_prisoners[color]];

    /* check whether we still live (i.e. detect suicide) */
    if (m_libs[m_parent[i]] == 0) {
        assert(captured_stones == 0);
        remove_string(i);
        return 0;
    }

    /* check for possible simple ko */
    if (captured_stones == 1 && m_libs[m_parent[i]] == 1 && m_stones[m_parent[i]] == 1) {
        return captured_sq;
    }

    // No ko
    return 0;
}


std::uint64_t FastBoard::test_update_ko_hash(const int color, const int i) const {

    // must NOT be suicide

    assert(i != FastBoard::PASS);
    assert(m_square[i] == EMPTY);

    auto ko_hash = m_ko_hash;

    ko_hash ^= Zobrist::zobrist[EMPTY][i];
    ko_hash ^= Zobrist::zobrist[color][i];

    std::array<int, 4> nbr_pars;
    int nbr_par_cnt = 0;

    for (auto ai : NEIGHBORS[i]) {

        if (m_square[ai] == !color) {
            if (m_libs[m_parent[ai]] == 1) {
                bool found = false;
                for (int n = 0; n < nbr_par_cnt; n++) {
                    if (nbr_pars[n] == m_parent[ai]) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    nbr_pars[nbr_par_cnt++] = m_parent[ai];
                }
            }
        }
    }

    for (int n = 0; n < nbr_par_cnt; n++) {
        int ai = nbr_pars[n];
        int pos = ai;

        do {
            ko_hash ^= Zobrist::zobrist[!color][pos];
            ko_hash ^= Zobrist::zobrist[EMPTY][pos];

            pos = m_next[pos];
        } while (pos != ai);
    }

    return ko_hash;
}

int FastBoard::text_to_move(const std::string& vertex) {

    if (vertex == "pass" || vertex == "PASS") return PASS;
    else if (vertex == "resign" || vertex == "RESIGN") return RESIGN;

    if (vertex.size() < 2) return -10;
    if (!std::isalpha(vertex[0])) return -11;
    if (!std::isdigit(vertex[1])) return -12;
    if (vertex[0] == 'i') return -13;

    int column, row;
    if (vertex[0] >= 'A' && vertex[0] <= 'Z') {
        if (vertex[0] < 'I') {
            column = vertex[0] - 'A';
        } else {
            column = (vertex[0] - 'A')-1;
        }
    } else {
        if (vertex[0] < 'i') {
            column = vertex[0] - 'a';
        } else {
            column = (vertex[0] - 'a')-1;
        }
    }

    std::string rowstring(vertex);
    rowstring.erase(0, 1);
    std::istringstream parsestream(rowstring);

    parsestream >> row;
    row--;

    if (row >= FastBoard::BOARDSIZE || column >= FastBoard::BOARDSIZE) {
        return -14;
    }

    auto move = row * BOARDSIZE + column;
    return move;
}
