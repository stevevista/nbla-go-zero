#include "gtp.h"
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>  
#include <stdarg.h>
#include <cstdlib>
#include <cctype>
#include "Leela.h"
#include "aq.h"



using namespace std;

std::pair<int, int> movetext2xy(const std::string& text) {

	if (text == "pass")
		return {-1, 0};
	
	if (text == "resign")
		return {-2, 0};
				
	static string x_list = "ABCDEFGHJKLMNOPQRSTabcdefghjklmnopqrst";

	std::string str_x = text.substr(0, 1);
	std::string str_y = text.substr(1);

	int x = int(x_list.find(str_x)) % 19;
	int y = std::stoi(str_y) - 1;

	return {x, y};
}

string xy2text(int x, int y) {

	static const char str_x[] = "ABCDEFGHJKLMNOPQRST";
	
	if(x == -1) return "pass";
	else if(x == -2) return "resign";

	string movetext;
	movetext = str_x[x];
	movetext += std::to_string(y+1);
	return movetext;
}


const string gtp_commands[] = {
    "protocol_version",
    "name",
    "version",
    "quit",
    "known_command",
    "list_commands",
    "quit",
    "boardsize",
    "clear_board",
    "komi",
    "play",
    "genmove",
    "showboard",
    "undo",
    "final_score",
    "final_status_list",
    "time_settings",
    "time_left",
    "fixed_handicap",
    "place_free_handicap",
    "set_free_handicap",
    "loadsgf",
    "printsgf",
    "kgs-genmove_cleanup",
    "kgs-time_settings",
    "kgs-game_over",
    "heatmap",
    ""
};


Gtp::Gtp(IGtpAgent* agent)
: agent_(agent)
{
	use_pondering_ = true;
}

void Gtp::enable_ponder(bool v) {
	use_pondering_ = v;
}

void Gtp::play(bool is_black, int x, int y) {

	string cmd = "play ";
	cmd += is_black ? "b " : "w ";
	cmd += xy2text(x, y);

	send_command(cmd);
}

void Gtp::genmove(bool is_black, bool commmit) {

	string cmd = "genmove ";
	cmd += is_black ? "b" : "w";
	if (!commmit)
		cmd += " nocommit";

	send_command(cmd);
}

void Gtp::stop_thinking() {
	if (agent_)
		agent_->stop_ponder();
}

void Gtp::run() {
	
	th_ = std::thread([&]() {
	
		CallGTP();
	});
}

int Gtp::CallGTP() {

	string command;
	bool is_playing = false;
	int id;

	bool black_playing = true;

	auto gtp_vprint = [&](string prefix, const char *fmt, va_list ap) {
		
		char buffer[4096];

		if (id != -1) {
			prefix += std::to_string(id);
		}

		int n1 = sprintf(buffer, "%s ", prefix.c_str());
		int n2 = vsprintf(&buffer[n1], fmt, ap);
		s_queue.push({command, string(buffer, n1+n2)});
	};

	auto gtp_print = [&](const char *fmt, ...) {
		va_list ap;
		va_start(ap, fmt);
		gtp_vprint("=", fmt, ap);
		va_end(ap);
	};

	auto gtp_fail = [&](const char *fmt, ...) {
		va_list ap;
		va_start(ap, fmt);
		gtp_vprint("?", fmt, ap);
		va_end(ap);
	};

	auto genmove = [&](bool commit) {

		is_playing = true;
		
		auto move = agent_->genmove(black_playing, commit);
		if (commit)
			black_playing = !black_playing;

		return xy2text(move.first, move.second);
	};

	auto play = [&](const string& movetext) {

		if (movetext == "pass")
			agent_->pass(black_playing);
		else if (movetext == "resign") {
			agent_->resign(black_playing);
			is_playing = false;
		} else {
			auto xy = movetext2xy(movetext);
			agent_->play(black_playing, xy.first, xy.second);
		}

		black_playing = !black_playing;
	};

	auto check_player = [&](const string& color) {

		bool is_black = std::toupper(color[0]) == 'B';
		
		if (black_playing != is_black) {
			agent_->pass(black_playing);
			black_playing = !black_playing;
		}
	};
		

	//    Start communication with the GTP protocol.
	for (;;) {
		id = -1;

		// Thread that monitors GTP commands during pondering.
		std::thread read_th([&] {
			string input_line;
			while(input_line == "" || input_line == "#") {
				
				r_queue.wait_and_pop(input_line);
			}

			agent_->stop_ponder();

			if (std::isdigit(input_line[0])) {
				std::istringstream strm(input_line);
				char spacer;
				strm >> id;
				strm >> std::noskipws >> spacer;
				std::getline(strm, command);
			} else {
				command = input_line;
			}
		});

		if (is_playing && use_pondering_)
			agent_->ponder_on_idle();

		read_th.join();

		if (use_pondering_)
			agent_->ponder_enable();

		// Process GTP command.
		if (command == "" || command == "\n") {
			continue;
		}
		else if (command.find("name") == 0) gtp_print("%s", agent_->name().c_str());
		else if (command.find("protocol_version") == 0) gtp_print("2.0");
		else if (command.find("version") == 0) gtp_print("2.0.3");
		else if (command.find("boardsize") == 0) {
			// Board size setting. (only corresponding to 19 size)
			// "=boardsize 19", "=boardsize 13", ...
			std::istringstream cmdstream(command);
			std::string stmp;
			int tmp;

			cmdstream >> stmp;  // eat boardsize
			cmdstream >> tmp;

			if (tmp != 19) {
				gtp_fail("unacceptable size");
			} else {
				agent_->clear_board();
				is_playing = false;
				black_playing = true;
			
				gtp_print("");
			}
		}
		else if (command.find("list_commands") == 0) {
			string outtmp(gtp_commands[0]);
			for (int i = 1; gtp_commands[i].size() > 0; i++) {
				outtmp = outtmp + "\n" + gtp_commands[i];
			}
			gtp_print(outtmp.c_str());
		}
		else if (command.find("clear_board") == 0)
		{
			agent_->clear_board();
			is_playing = false;
			black_playing = true;

			gtp_print("");
			std::cerr << "clear board." << std::endl;
		}
		else if (command.find("komi") == 0)
		{
			std::istringstream cmdstream(command);
			std::string tmp;
			float komi = 7.5f;
	
			cmdstream >> tmp;  // eat komi
			cmdstream >> komi;
	
			if (!cmdstream.fail()) {
				agent_->komi(komi);
				gtp_print("");
			} else {
				gtp_fail("syntax not understood");
			}
		}
		else if (command.find("time_left") == 0)
		{
			//
			// Set remaining time.
			// "=time_left B 944", "=time_left white 300", ...
			std::istringstream cmdstream(command);
			std::string tmp, color;
			int time, stones;
	
			cmdstream >> tmp >> color >> time >> stones;
	
			if (!cmdstream.fail()) {
				int icolor;
	
				if (color == "w" || color == "white") {
					icolor = 0;
				} else if (color == "b" || color == "black") {
					icolor = 1;
				} else {
					gtp_fail("Color in time adjust not understood.\n");
					continue;
				}
	
				agent_->time_left(icolor, time);
	
				gtp_print("");
			} else {
				gtp_fail("syntax not understood");
			}
		}
		else if (command.find("go") == 0) {

			auto move = genmove(true);
			gtp_print("%s", move.c_str());
		} 
		else if (command.find("auto") == 0) {
			int passes = 0;
			int move;
			do {
				auto move = genmove(true);
				std::cerr << "> " << move << std::endl;
				if (move == "pass") passes++;
				else if (move == "resign") break;
				else passes = 0;

			} while (passes < 2 && move != -2);
			gtp_print("");
		} 
		else if (command.find("genmove") == 0) {

			std::istringstream cmdstream(command);
			std::string tmp;
	
			cmdstream >> tmp;  // eat genmove
			cmdstream >> tmp;

			check_player(tmp);
			bool commit = command.find("nocommit") == std::string::npos;

			auto move = genmove(commit);
			gtp_print("%s", move.c_str());
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

			check_player(color);
			play(vertex);
			gtp_print("");
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
			gtp_print("");
		}
		else if(command.find("final_score") == 0) {

			auto final_score = agent_->final_score();

			// c. Send GTP response.
			std::string win_pl = final_score > 0 ? "B+" : "W+";
			std::stringstream ss;
			ss << std::fixed << std::setprecision(1) << final_score;
			win_pl += ss.str();
			if(final_score == 0) win_pl = "0";

			gtp_print("%s", win_pl.c_str());
		}
		else if(command.find("isready") == 0) {
			gtp_print("readyok");
		}
		else if(command.find("ponder") == 0) {
			is_playing = true;
			gtp_print("ponder started.");
		}
		else if (command.find("time_settings") == 0) {
			std::istringstream cmdstream(command);
			std::string tmp;
			int maintime, byotime, byostones;

			cmdstream >> tmp >> maintime >> byotime >> byostones;

			if (!cmdstream.fail()) {
				// convert to centiseconds and set
				agent_->set_timecontrol(maintime, byotime, byostones, 0);

				gtp_print("");
			} else {
				gtp_fail("syntax not understood");
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
				gtp_print("");
			} else {
				gtp_fail("syntax not understood");
			}
		}
		else if(command.find("kgs-game_over") == 0){
			agent_->game_over();
			is_playing = false;
			gtp_print("");
		}
		else if(command == "quit") {
			agent_->quit();
			gtp_print("");
			break;
		}
		else{
			gtp_print("");
			std::cerr << "unknown command.\n";
		}
	}

	return 0;

}


void play_matchs(const std::string& sgffile, IGtpAgent* player1, IGtpAgent* player2, std::function<void(int, int[])> callback) {

	player1->clear_board();
	player2->clear_board();

	std::pair<int, int> xy1 = {0, 0};
	std::pair<int, int> xy2 = {0, 0};
	int move_cnt = 0;
	int winner = -1;
	
	int board[361];
    auto fill_board = [&]() {
        for (int i=0; i<361; i++) {
            board[i] = player1->get_color(i);
        }
    };

	while (true) {
		xy1 = player1->genmove(1, true);
		if (xy1.first == -1) player2->pass(1);
		else if (xy1.first == -2) player2->resign(1);
		else player2->play(true, xy1.first, xy1.second);

		move_cnt++;

		if (callback) {
			fill_board();
            int idx;
            if (xy1.first >= 0) {
                idx = xy1.second*19 + xy1.first;
			} else
				idx = xy1.first;
            callback(idx, board);
		}

		if (xy1.first == -2)
			break;
		if (xy2.first == -1 && xy1.first == -1)
			break;

		xy2 = player2->genmove(false, true);
		if (xy2.first == -1) player1->pass(0);
		else if (xy2.first == -2) player1->resign(0);
		else player1->play(false, xy2.first, xy2.second);

		move_cnt++;

		if (xy2.first == -2)
			break;
		if (xy2.first == -1 && xy1.first == -1)
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



////////////////////////////////////////////////////////////

std::shared_ptr<IGtpAgent> create_agent(const std::vector<std::string>& args) {

	std::string engine_type;
	
	for (auto i = 0; i < args.size(); i++) { 
        auto opt = args[i];
		
		if (opt == "--engine_type") {
			engine_type = args[++i];
		}
	}

	if (engine_type == "aq")
		return std::make_shared<AQ>(args);
	else if (engine_type == "policy")
		return std::make_shared<PolicyPlayer>(args);
	else
		return std::make_shared<Leela>(args);
}

