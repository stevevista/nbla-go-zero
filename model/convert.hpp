#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "Board.h"

using std::vector;


using BoardPlane = std::bitset<361>;
using InputFeature = std::vector<BoardPlane>;

struct MoveData {
    InputFeature input;
    std::vector<float> probs;
    int result;
};

class GameArchive {
    struct game_t {
        vector<short> seqs;
        std::vector<std::array<float, 362>> dists;
        int result;
    };
    vector<game_t> games_;
    vector<std::pair<int, int>> entries_; // game_index + step_index
    bool follow_distribution_;

    void extract_move(const int index, MoveData& out);

public:
    int data_index;
    const int boardsize = 19;

    static void generate(const std::string& path, const std::string& out_path, int random_drop);

    GameArchive();
    void shuffle();
    int total_moves() const { return entries_.size(); }

    int load(const std::string& path, bool append);

    vector<MoveData> next_batch(int count, bool& rewinded);
};
