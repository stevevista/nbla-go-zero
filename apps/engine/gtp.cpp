#include "gtp.h"
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>  
#include <stdarg.h>
#include <cstdlib>
#include <cctype>
#include "leela/Utils.h"
#include "aq/board_config.h"

using namespace Utils;

static bool FindStr(std::string str, std::string s1){
	return 	str.find(s1) != std::string::npos;
}
static bool FindStr(std::string str, std::string s1, std::string s2){
	return 	str.find(s1) != std::string::npos ||
			str.find(s2) != std::string::npos;
}
static bool FindStr(std::string str, std::string s1, std::string s2, std::string s3){
	return 	str.find(s1) != std::string::npos ||
			str.find(s2) != std::string::npos ||
			str.find(s3) != std::string::npos;
}
static bool FindStr(std::string str, std::string s1, std::string s2, std::string s3, std::string s4){
	return 	str.find(s1) != std::string::npos ||
			str.find(s2) != std::string::npos ||
			str.find(s3) != std::string::npos ||
			str.find(s4) != std::string::npos;
}




/**
 *  Break character string with delimiter.
 */
template <typename List>
void SplitString(const std::string& str, const std::string& delim, List& split_list)
{
    split_list.clear();

    using string = std::string;
    string::size_type pos = 0;

    while(pos != string::npos ){
        string::size_type p = str.find(delim, pos);

        if(p == string::npos){
        	split_list.push_back(str.substr(pos)); break;
        }
        else split_list.push_back(str.substr(pos, p - pos));

        pos = p + delim.size();
    }
}


std::string xy2movetext(int x, int y) {

	std::string str_v;
	std::string str_x = "ABCDEFGHJKLMNOPQRST";
	str_v = str_x[x - 1];
	str_v += std::to_string(y);
	return str_v;
}

std::pair<int, int> movetext2xy(const std::string& text) {

	if (text == "pass")
		return {19, 18};
	
	if (text == "resign")
		return {20, 18};
				
	std::string x_list = "ABCDEFGHJKLMNOPQRSTabcdefghjklmnopqrst";

	std::string str_x = text.substr(0, 1);
	std::string str_y = text.substr(1);

	int x = int(x_list.find(str_x)) % 19;
	int y = std::stoi(str_y) - 1;

	return {x, y};
}

static int movetext2extend(const std::string& text) {
	auto xy = movetext2xy(text);
	int x = xy.first;
	int y = xy.second;

	return xytoe[x+1][y+1];
}


std::string CoordinateString(int v);


Gtp::Gtp(IGtpAgent* agent)
: agent_(agent)
{
	use_pondering_ = true;
}

void Gtp::enable_ponder(bool v) {
	use_pondering_ = v;
}

int Gtp::CallGTP() {

	std::string gtp_str;
	std::string command;
	std::vector<std::string> split_list;
	bool is_playing = false;
	int id = -1;

	auto SendGTP = [&](const char* output_str, ...) {
		char buffer[4096];
		va_list args;
		va_start(args, output_str);
		int n = vsprintf(buffer, output_str, args);
		va_end(args);

		s_queue.push({gtp_str, std::string(buffer, n)});
	};

	//    Start communication with the GTP protocol.
	for (;;) {
		gtp_str = "";

		// Thread that monitors GTP commands during pondering.
		std::thread read_th([&] {
			while(gtp_str == "" || gtp_str == "#") {
				
				r_queue.wait_and_pop(gtp_str);
			}

			agent_->stop_ponder();

			if (std::isdigit(gtp_str[0])) {
				std::istringstream strm(gtp_str);
				char spacer;
				strm >> id;
				strm >> std::noskipws >> spacer;
				std::getline(strm, command);
			} else {
				command = gtp_str;
			}
		});

		if (is_playing && use_pondering_)
			agent_->ponder_on_idle();

		read_th.join();

		if (use_pondering_)
			agent_->ponder_enable();

		// Process GTP command.
		if (gtp_str == "" || gtp_str == "\n") {
			continue;
		}
		else if (command.find("name") == 0) SendGTP("= %s\n\n", agent_->name().c_str());
		else if (command.find("protocol_version") == 0) SendGTP("= 2.0\n\n");
		else if (command.find("version") == 0) SendGTP("= 2.0.3\n\n");
		else if (command.find("boardsize") == 0) {
			// Board size setting. (only corresponding to 19 size)
			// "=boardsize 19", "=boardsize 13", ...
			std::istringstream cmdstream(command);
			std::string stmp;
			int tmp;

			cmdstream >> stmp;  // eat boardsize
			cmdstream >> tmp;

			if (tmp != 19) {
				SendGTP("unacceptable size\n\n");
			} else {
				agent_->clear_board();
				is_playing = false;
				SendGTP("= \n\n");
			}
		}
		else if (FindStr(gtp_str, "list_commands"))
		{
			// Send the corresponding command list.
			const char* commands = "= boardsize\n"
			"list_commands\n"
			"clear_board\n"
			"genmove\n"
			"play\n"
			"quit\n"
			"time_left\n"
			"time_settings\n"
			"name\n"
			"protocol_version\n"
			"version\n"
			"komi\n"
			"final_score\n"
			"kgs-time_settings\n"
			"kgs-game_over\n"
			"= \n\n";
			SendGTP(commands);
		}
		else if (command.find("clear_board") == 0)
		{
			agent_->clear_board();
			is_playing = false;

			SendGTP("= \n\n");
			std::cerr << "clear board." << std::endl;
		}
		else if (FindStr(gtp_str, "komi"))
		{
			SplitString(gtp_str, " ", split_list);
			if(split_list[0] == "=") split_list.erase(split_list.begin());

			double komi_ = stod(split_list[1]);
			agent_->komi(komi_);

			SendGTP("= \n\n");
		}
		else if (FindStr(gtp_str, "time_left"))
		{
			//
			// Set remaining time.
			// "=time_left B 944", "=time_left white 300", ...
			SplitString(gtp_str, " ", split_list);
			if(split_list[0] == "=") split_list.erase(split_list.begin());

			int left_time = stoi(split_list[2]);
			int pl = FindStr(gtp_str, "B", "b")? 1 : 0;
			agent_->time_left(pl, left_time);

			SendGTP("= \n\n");
		}
		else if (command.find("go") == 0) {

			int next_move = agent_->genmove();
			if(next_move == -1) {
				SendGTP("= pass\n\n");
			} else if (next_move == -2) {
				SendGTP("= resign\n\n");
			} else{
				std::string str_nv = CoordinateString(next_move);
				SendGTP("= %s\n\n", str_nv.c_str());
			}
		} 
		else if (command.find("auto") == 0) {
			int passes = 0;
			int move;
			do {
				move = agent_->genmove();
				std::string str_nv = CoordinateString(move);
				std::cerr << "> " << str_nv << std::endl;
				if (move == -1) passes++;
				else passes = 0;

			} while (passes < 2 && move != -2);
			SendGTP("= \n\n");
		} 
		else if (FindStr(gtp_str, "genmove")) {

			// Think and send the next move.
			// "=genmove b", "=genmove white", ...
			std::istringstream cmdstream(command);
			std::string tmp;
	
			cmdstream >> tmp;  // eat genmove
			cmdstream >> tmp;

			int pl = std::toupper(tmp[0]) == 'B' ? 1 : 0;
			bool commit = true;
			if (command.find("nocommit") != std::string::npos) {
				commit = false;
			}

			is_playing = true;
			int next_move = agent_->genmove(pl, commit);
			// d. Send response of the next move.
			if(next_move == -1) {
				SendGTP("= pass\n\n");
			} else if (next_move == -2) {
				SendGTP("= resign\n\n");
			} else{
				std::string str_nv = CoordinateString(next_move);
				SendGTP("= %s\n\n", str_nv.c_str());
			}
		}
		else if (command.find("play") == 0)
		{
			//
			// Receive the opponent's move and reflect on the board.
			// "=play w D4", "play b pass", ...

			std::istringstream cmdstream(command);
			std::string tmp;
			std::string color, vertex;

			cmdstream >> tmp;   //eat play
			cmdstream >> color;
			cmdstream >> vertex;

			int pl = std::toupper(color[0]) == 'B' ? 1 : 0;

			int next_move;
			if (command.find("pass") != std::string::npos) {
				agent_->pass(pl);
			} else if (command.find("resign") != std::string::npos) {
				agent_->resign(pl);
				is_playing = false;
			} else {
				next_move = movetext2extend(vertex);
				agent_->play(pl, next_move);
			}

			// d. Send GTP response.
			SendGTP("= \n\n");
		}
		else if (command.find("heatmap") == 0) {
			std::istringstream cmdstream(command);
			std::string tmp;
			int rotation;
	
			cmdstream >> tmp;   // eat heatmap
			cmdstream >> rotation;
	
			if (!cmdstream.fail()) {
				agent_->heatmap(rotation);
			} else {
				agent_->heatmap(0);
			}
			SendGTP("= \n\n");
		}
		else if(FindStr(gtp_str, "final_score")) {

			auto final_score = agent_->final_score();

			// c. Send GTP response.
			std::string win_pl = final_score > 0 ? "B+" : "W+";
			std::stringstream ss;
			ss << std::fixed << std::setprecision(1) << final_score;
			win_pl += ss.str();
			if(final_score == 0) win_pl = "0";

			SendGTP("= %s\n\n", win_pl.c_str());
		}
		else if(FindStr(gtp_str, "isready")) {
			SendGTP("= readyok\n");
		}
		else if(FindStr(gtp_str, "ponder")) {
			is_playing = true;
			SendGTP("= ponder started.\n");
		}
		else if (command.find("time_settings") == 0) {
			std::istringstream cmdstream(command);
			std::string tmp;
			int maintime, byotime, byostones;

			cmdstream >> tmp >> maintime >> byotime >> byostones;

			if (!cmdstream.fail()) {
				// convert to centiseconds and set
				agent_->set_timecontrol(maintime, byotime, byostones, 0);

				SendGTP("= \n\n");
			} else {
				SendGTP("syntax not understood\n\n");
			}
		}
		else if (command.find("kgs-time_settings") == 0) {
			// none, absolute, byoyomi, or canadian
			std::istringstream cmdstream(command);
			std::string tmp;
			std::string tc_type;
			int maintime, byotime, byostones, byoperiods;

			cmdstream >> tmp >> tc_type;

			if (tc_type.find("none") != std::string::npos) {
				// 30 mins
				agent_->set_timecontrol(30 * 60, 0, 0, 0);
			} else if (tc_type.find("absolute") != std::string::npos) {
				cmdstream >> maintime;
				agent_->set_timecontrol(maintime, 0, 0, 0);
			} else if (tc_type.find("canadian") != std::string::npos) {
				cmdstream >> maintime >> byotime >> byostones;
				// convert to centiseconds and set
				agent_->set_timecontrol(maintime, byotime, byostones, 0);
			} else if (tc_type.find("byoyomi") != std::string::npos) {
				// KGS style Fischer clock
				cmdstream >> maintime >> byotime >> byoperiods;
				agent_->set_timecontrol(maintime, byotime, 0, byoperiods);
			}

			if (!cmdstream.fail()) {
				SendGTP("= \n\n");
			} else {
				SendGTP("syntax not understood\n\n");
			}
		}
		else if(FindStr(gtp_str, "kgs-game_over")){
			agent_->game_over();
			is_playing = false;
			SendGTP("= \n\n");
		}
		else if(FindStr(gtp_str, "quit")){
			agent_->quit();
			SendGTP("= \n\n");
			break;
		}
		else{
			SendGTP("= \n\n");
			std::cerr << "unknown command.\n";
		}
	}

	return 0;

}


void play_matchs(const std::string& sgffile, IGtpAgent* player1, IGtpAgent* player2, std::function<void(int, int[])> callback) {

	player1->clear_board();
	player2->clear_board();

	int move1 = 0;
	int move2 = 0;
	int move_cnt = 0;
	int winner = -1;
	
	int board[361];
    auto fill_board = [&]() {
        for (int i=0; i<361; i++) {
            board[i] = player1->get_color(i);
        }
    };

	while (true) {
		move1 = player1->genmove(1, true);
		if (move1 == -1) player2->pass(1);
		else if (move1 == -2) player2->resign(1);
		else player2->play(1, move1);

		move_cnt++;

		if (callback) {
			fill_board();
            int idx = move1;
            if (move1 >= 0) {
                idx = etor[move1];
            }
            callback(idx, board);
		}

		if (move1 == -2)
			break;
		if (move2 == -1 && move1 == -1)
			break;

		move2 = player2->genmove(0, true);
		if (move2 == -1) player1->pass(0);
		else if (move1 == -2) player1->resign(0);
		else player1->play(0, move1);

		move_cnt++;

		if (move2 == -2)
			break;
		if (move2 == -1 && move1 == -1)
			break;

		if (move_cnt >= 361*2)
            break;
	}

	// Nobody resigned, we will have to count
    if (winner == -1) {
        float score = player1->final_score();
        if (score < 0) {
            winner = 0;
        } else if (score > 0) {
            winner = 1;
        }
    }
    
    int result = 0;
    if (winner == 1)
        result = 1;
    else if (winner == 0)
        result = -1;
    else
        result = 0;

	if (sgffile.size()) {
		if (!player1->dump_sgf(sgffile))
			player2->dump_sgf(sgffile);
	}

    
    std::cerr << "final score: " << result << std::endl;
}
