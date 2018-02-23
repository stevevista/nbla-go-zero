


#include <model/convert.hpp>
#include <algorithm>

static std::string opt_input = "../../data/sgfs/train";
static std::string opt_output = "../../data/train.data";

void parse_commandline( int argc, char **argv ) {
    for (int i = 1; i < argc; i++) { 
        std::string opt = argv[i];

        if (opt == "--input" || opt == "-i") {
            opt_input = argv[++i];
        } else if (opt == "--output" || opt == "-o") {
            opt_output = argv[++i];
        }
    }
}

int main(int argc, char **argv) {

    parse_commandline(argc, argv);
    GoBoard::init_board();
    GameArchive::generate(opt_input, opt_output, 6);
    return 0;
}
