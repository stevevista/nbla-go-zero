


#include <model/convert.hpp>
#include <algorithm>


const std::string command[] = {
    "--input",
    "--output",
    ""
};


static std::string opt_input = "../../data/sgfs/train";
static std::string opt_output = "../../data/train.data";

void parse_commandline( int argc, char **argv ) {
    for (int i = 1; i < argc; i++) { 
        std::string opt;
        for (int j = 0; command[j].size(); j++){
            if (command[j] == argv[i]){
                opt = command[j];
                break;
            }
        }

        if (opt == "--input") {
            opt_input = argv[++i];
        } else if (opt == "--output") {
            opt_output = argv[++i];
        }
    }
}

int main(int argc, char **argv) {

    parse_commandline(argc, argv);
    generate_data(opt_input, opt_output, 6);
    return 0;
}
