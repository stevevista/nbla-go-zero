
#include <engine/leela/GameState.h>
#include <engine/aq/aq.h>
#include <engine/leela/Leela.h>
#include <iostream>
#include <sstream>
#include <cstdarg>
#include <cctype>
#include <simplemodel/zero_model.hpp>

#include <fstream>
#if defined (_WIN32)
#include <windows.h>
#endif


template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args)
    {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
    

static std::string cfg_logfile;
static std::string cfg_weightsfile = "../../data/leela.weights";
static std::string opt_model;
static bool cfg_ponder = true;


const std::string command[] = {
    "--e",
    "--w",
    "--ponder",
    "--log",
    ""
  };
  
void parse_commandline( int argc, char **argv) {
      for (int i = 1; i < argc; i++) { 
          std::string opt;
          for (int j = 0; command[j].size(); j++){
              if (command[j] == argv[i]){
                  opt = command[j];
                  break;
              }
          }
  
        if (opt == "--e") {
              opt_model = argv[++i];
        } else if (opt == "--w") {
              cfg_weightsfile = argv[++i];
        } else if (opt == "--ponder") {
              cfg_ponder = std::string(argv[++i]) != "off";
        } else if (opt == "--log") {
            cfg_logfile = argv[++i];
        }
    }
}




int main(int argc, char *argv[]) {

    parse_commandline(argc, argv);

    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
#ifndef WIN32
    setbuf(stdin, NULL);
#endif

    std::unique_ptr<ZeroPredictModel> model;
    std::unique_ptr<IGtpAgent> agent;

    model = std::unique_ptr<ZeroPredictModel>(new ZeroPredictModel());
    model->load_weights(cfg_weightsfile);

    if (opt_model == "aq") {
        agent = std::unique_ptr<AQ>(new AQ(model.get(), cfg_weightsfile));
        std::cerr << "configuration loaded.\n";
    } else if (opt_model == "policy") {
        agent = std::unique_ptr<PolicyPlayer>(new PolicyPlayer(model.get(), cfg_logfile));
    } else {
        agent = std::unique_ptr<Leela>(new Leela(model.get(), cfg_logfile));
    }

    
	Gtp gtp(agent.get());

    gtp.enable_ponder(cfg_ponder);
	gtp.run();

	std::thread([&]() {

		while(true) {
			std::string input_str;
			getline(std::cin, input_str);
			if (input_str.find("stop")==0) {
				//gtp.stop_thinking();
				continue;
			}
			gtp.send_command(input_str);
			if (input_str == "quit")
				break;
		}

	}).detach();

	std::thread([&](){

		while (true) {
			std::string prev_cmd, rsp;
			gtp.unsolicite(prev_cmd, rsp);
			
			std::cout << rsp << std::endl;
			if (prev_cmd == "quit")
				break;
		}

	}).detach();

	//DoSomething();

	gtp.join();
	std::cerr << "finished.\n";


    return 0;
}
