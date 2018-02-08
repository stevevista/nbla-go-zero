#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "Board.h"

using std::vector;

void generate_data(const std::string& path, const std::string& out_path, int random_drop);

using PLANES = InputFeature;
using PROBS =  std::vector<float>;

struct MoveData {
    PLANES planes;
    PROBS probs;
    int result;
};

class GameArchive {
    struct game_t {
        vector<short> moves;
        int result;
    };
    vector<game_t> games_;
    vector<std::pair<int, int>> entries_; // game_index + step_index

    void extract_move(const int index, PLANES& planes, PROBS& probs, int& result);

public:
    int data_index;
    const int boardsize = 19;

    GameArchive();
    void shuffle();
    int total_moves() const { return entries_.size(); }

    int load(const std::string& path, bool append);
    void add(const vector<short>& moves, int result);
    void add(const vector<int>& moves, int result);

    vector<MoveData> next_batch(int count, bool fitsize, bool& rewinded, int min_steps=-1);
};
