#include <stdio.h>
#include <stdarg.h>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <iomanip>
#include <algorithm>  

#include "gtp.h"
#include "aq.h"

using std::string;
using std::cerr;
using std::endl;


bool need_time_controll = false;


#if defined (_WIN32)
#define PATH_SEP '\\'
#else
#define PATH_SEP '/'
#endif


inline void trim(string &ss)   
{   
	auto p=find_if(ss.rbegin(),ss.rend(),std::not1(std::ptr_fun(isspace)));   
    ss.erase(p.base(),ss.end());  
	auto p2 = std::find_if(ss.begin(),ss.end(),std::not1(std::ptr_fun(isspace)));   
	ss.erase(ss.begin(), p2);
} 


AQ::AQ(const std::vector<std::string>& args) {
	
	for (auto i = 0; i < args.size(); i++) { 
        auto opt = args[i];

		if (opt == "--working_dir") {
			working_dir = args[++i];
        } else if (opt == "--thread_cnt") {
			cfg_thread_cnt = std::stoi(args[++i]);
        } else if (opt == "--byoyomi") {
            cfg_byoyomi 	= std::stod(args[++i]); 
        } else if (opt == "--main_time") {
            cfg_main_time 	= std::stod(args[++i]); 
        } else if (opt == "--need_time_controll") {
			++i;
            need_time_controll = (args[i] == "true" || args[i] == "on");
        } else if (opt == "--japanese_rule") {
            ++i;
			japanese_rule 	= (args[i] == "true" || args[i] == "on");
        } else if (opt == "--komi") {
            cfg_komi 		= std::stod(args[++i]);
        } else if (opt == "--never_resign") {
			++i;
            never_resign 	= (args[i] == "true" || args[i] == "on");
        }
    }
	
	ImportProbDist();
	ImportProbPtn3x3();
}

void Gtp::run() {

    th_ = std::thread([&]() {

        CallGTP();
    });
}

void Gtp::stop_thinking() {
	if (agent_)
		agent_->stop_ponder();
}


void AQ::clear_board() {
	// Initialize the board.
	b.Clear();
	tree.InitBoard();
}

void AQ::komi(float v) {
	cfg_komi = (v == 0)? 0.5 : v;
	tree.komi = cfg_komi;
}

int AQ::genmove() {
	return genmove(b.my, true);
}

int AQ::genmove(int player, bool commit) {
	
	auto t1 = std::chrono::system_clock::now();
	cerr << "thinking...\n";

	if(player != b.my){
		// Insert pass if the turn is different.
		b.PlayLegal(PASS);
		tree.UpdateRootNode(b);
		--b.pass_cnt[b.her];
	}

	int next_move;
	tree.stop_think = false;
	bool think_full = true;
	double win_rate;

	if(think_full) {
		//    Search for the best move.
		next_move = tree.SearchTree(b, 0.0, win_rate, true, false);
	}

	else if(win_rate < 0.1) {
				
		// Roll out 1000 times to check if really losing.
		Board b_;
		int win_cnt = 0;
		for(int i=0;i<1000;++i){
					b_ = b;
					int result = PlayoutLGR(b_, tree.lgr, tree.komi);
					if(b.my == std::abs(result)) ++win_cnt;
		}
		if((double)win_cnt / 1000 < 0.25) next_move = PASS;
	}

	if (commit) {
		// c. Play the move.
		b.PlayLegal(next_move);
		tree.UpdateRootNode(b);
	}

	if(next_move == PASS){
		if(!never_resign && win_rate < 0.1) next_move = -2; // RESIGN
		else next_move = -1; //PASS
	}

	// g. Update remaining time.
	if(need_time_controll){
		auto t2 = std::chrono::system_clock::now();
		double elapsed_time = (double)std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()/1000;
		tree.left_time = std::max(0.0, (double)tree.left_time - elapsed_time);
	}

	return next_move;
}

void AQ::pass(int player) {
    b.PlayLegal(PASS);
	tree.UpdateRootNode(b);
}
        
void AQ::resign(int player) {
    b.PlayLegal(PASS);
	tree.UpdateRootNode(b);
}

void AQ::stop_ponder() {
	tree.stop_think = true;
}

void AQ::ponder_on_idle() {
	// Ponder until GTP commands are sent.
	if(	b.prev_move[b.her] != PASS &&
			(tree.left_time > 25 || tree.byoyomi != 0))
	{
		double time_limit = 300.0;
		double win_rate;
		tree.SearchTree(b, time_limit, win_rate, false, true);
	}
}

void AQ::ponder_enable() {
	tree.stop_think = false;
}


void AQ::time_left(int player, double t) {
	if(b.my == player)
	{
		tree.left_time = t;
		std::fprintf(stderr, "left time: %d[sec]\n", (int)t);
	}
}

std::string AQ::name() {
	return "AQ";
}

void AQ::play(int player, int move) {

	//    Insert pass before placing a opponent's stone.
	if (b.my != player)
	{
		b.PlayLegal(PASS);
		--b.pass_cnt[b.her];
	}

	// c. Play the move.
	b.PlayLegal(move);
	tree.UpdateRootNode(b);
}

float AQ::final_score() {
	// 
	// Roll out 1000 times and return the final score.

	// a. Roll out 1000 times.
	tree.stat.Clear();
	int win_cnt = 0;
	int rollout_cnt = 1000;
	for(int i=0;i<rollout_cnt;++i){
		Board b_cpy = b;
		int result = PlayoutLGR(b_cpy, tree.lgr, tree.stat, tree.komi);
		if(result != 0) ++win_cnt;
	}
	bool is_black_win = ((float)win_cnt/rollout_cnt >= 0.5);

	// b. Calculate scores in Chinese rule.
	float score[2] = {0.0, 0.0};
	for(int v=0;v<BVCNT;++v){
		if((float)tree.stat.owner[0][v]/tree.stat.game[2] > 0.5){
			++score[0];
		}
		else ++score[1];
	}
	float final_score = std::abs(score[1] - score[0] - tree.komi);

	if (!is_black_win)
		final_score = -final_score;
			
	return final_score;
}

void AQ::set_timecontrol(int maintime, int byotime, int byostones, int byoperiods) {
	tree.main_time = maintime;
	tree.left_time = tree.main_time;
	tree.byoyomi = byotime;
}

void AQ::quit() {
}

void AQ::game_over() {
	
}

void AQ::heatmap(int rotation) const {
	
}

int AQ::get_color(int idx) const {
	int c = b.color[rtoe[idx]];
	// empty->0, outer boundary->1, white->2, black->3
	if (c == 3) return 1;
	else if (c==2) return 2;
	else return 0;
}

bool AQ::dump_sgf(const std::string& path) const {
	
	return false;
}

