#pragma once

#include "AQ/src/board.h"
#include "AQ/src/search.h"
#include "AQ/src/nueral_net.h"
#include "gtp.h"

class AQ : public IGtpAgent {
    
    Board b;
    std::shared_ptr<Tree> tree;
    
public:
    AQ() = delete;
    AQ(const std::vector<std::string>& args);
    
    std::string name();
        void clear_board();
        void komi(float);
        void time_left(int player, double t);
    void play(bool is_black, int x, int y);
        void pass(int player);
        void resign(int player);
    std::pair<int, int> genmove(bool is_black, bool commit);
        void ponder_on_idle();
        void ponder_enable();
        void stop_ponder();
        float final_score();
    void set_timecontrol(int maintime, int byotime, int byostones, int byoperiods);

    void game_over();
    void quit();
    void heatmap(int rotation) const;
    int get_color(int idx) const;
    bool dump_sgf(const std::string& path) const;
};

    