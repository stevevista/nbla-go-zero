#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "Board.h"

using std::vector;



struct MoveData {
    InputFeature input;
    std::vector<float> probs;
    int result;
};

class GameArchive {
    struct game_t {
        vector<short> moves;
        int result;
    };
    vector<game_t> games_;
    vector<std::pair<int, int>> entries_; // game_index + step_index

    void extract_move(const int index, MoveData& out);

public:
    int data_index;
    const int boardsize = 19;

    static void generate(const std::string& path, const std::string& out_path, int random_drop);

    GameArchive();
    void shuffle();
    int total_moves() const { return entries_.size(); }

    int load(const std::string& path, bool append);
    void add(const vector<short>& moves, int result);
    void add(const vector<int>& moves, int result);

    vector<MoveData> next_batch(int count, bool& rewinded);
};
