#include <model/convert.hpp>
#include <model/utils.hpp>
#include "leela/nn.h"
#include <fstream>
#include <cassert>


using std::make_shared;

using namespace napp;
/*

float sec(clock_t clocks)
{
    return (float)clocks/CLOCKS_PER_SEC;
}

void verify(
    const std::string& data_path, 
    const std::string& weight_path, 
    const int batch_size) {

    GoBoard::init_board();
    
    GameArchive m;
    auto N = m.load(data_path, true);
    if (N == 0)
        throw std::runtime_error("cannot load training data");

    const int boardsize = m.boardsize;
    std::cout << "training data, boardsize = " << boardsize << std::endl;
    std::cout << "total samples = " << N << std::endl;

    zero_model model;
    model.load_weights(weight_path);

    size_t seen = 0;
    int batchs = 0;
    uint64_t total_count = 0;
    uint64_t correct_dist = 0;
    uint64_t correct_res = 0;

    while (true) {

        clock_t time=clock();

        bool rewinded;
        auto batch = m.next_batch(batch_size, rewinded);
        if (rewinded) {
            break;
        }

        for (const auto& d : batch) {

            auto output = model.predict(d.input);

            int move = max_index(d.probs);
            int predit_move = max_index(output.first.begin(), output.first.begin()+361);

           // if (move == 300)
            //    std::cout << predit_move << std::endl;
            
            int predict_res = (output.second > 0.0) ? 1 : -1;
            int res = d.result;

            total_count++;
            if (predit_move == move) correct_dist++;
            if (predict_res == res) correct_res++;
        }
        

        int acc_dist = (correct_dist*100) / total_count;
        int acc_res = (correct_res*100) / total_count;
        seen += batch_size;
        batchs++;

        if(batchs%10 == 0)
            printf("%ld, %.3f: %d%% predict, %d%% result, %lf seconds, %zu images\n", batchs, (float)seen/N, acc_dist, acc_res, sec(clock()-time), seen);

        if (rewinded)
            break;
    }

}


static std::string opt_weights = "../../data/leela.weights";
static std::string opt_input = "../../data/val.data";
static int opt_batch_size = 32;

void parse_commandline( int argc, char **argv ) {
    for (int i = 1; i < argc; i++) { 
        string opt = argv[i];

        if (opt == "--w" || opt == "--weights") {
            opt_weights = argv[++i];
        } else if (opt == "--input") {
            opt_input = argv[++i];
        } else if (opt == "--batch-size" || opt == "--b") {
            opt_batch_size = atoi(argv[++i]);
        }
    }
}
*/
void test_lite();

int main(int argc, char **argv) {

    test_lite();
    return 0;
/*
    parse_commandline(argc, argv);
    verify(
        opt_input, 
        opt_weights, 
        128);
    return 0;*/
}

