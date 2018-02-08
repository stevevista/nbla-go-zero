#pragma once

#include "board.h"
#include "search.h"
#include "nueral_net.h"
#include "../gtp.h"

class NENG_API AQ : public IGtpAgent {
    
        Board b;
        Tree tree;
    
    public:
        AQ(IPreditModel* m, const std::string& cfg_path);
    
        std::string name();
        void clear_board();
        void komi(float);
        void time_left(int player, double t);
        void play(int player, int move);
        void pass(int player);
        void resign(int player);
        int genmove(int player, bool commit);
        int genmove();
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

    