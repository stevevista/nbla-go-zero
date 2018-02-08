

#include <model/utils.hpp>
#include <nblapp/parameter.hpp>
#include <engine/leela/Leela.h>
#include <engine/leela/Training.h>
#include <model/zero_model.hpp>
#include <simplemodel/zero_model.hpp>

#include <fstream>
#include <cassert>
#include "dlib/ui.h"

using std::make_shared;
using namespace nblapp;
using namespace napp;


float sec(clock_t clocks)
{
    return (float)clocks/CLOCKS_PER_SEC;
}

void train(
    const std::string& weight_path, 
    const std::string& init_weight_path, 
    int batch_size, 
    float learning_rate,
    int game_rounds,
    int train_rounds,
    int playouts,
    int clean_per_rounds) {

    float weight_decay = 0.0001;


    ZeroTrainModel train_model(batch_size);
    train_model.load_weights(weight_path);

    batch_size = train_model.get_batch_size();
    if (learning_rate == 0) {
        learning_rate = train_model.get_learning_rate();
    }
    int seen = train_model.get_seen();
    int batchs = train_model.get_batchs();

    ZeroPredictModel play_model;

    std::cout << "batch_size = " << batch_size << std::endl;
    std::cout << "learning_rate = " << learning_rate << std::endl;


    auto solver = MomentumSolver(learning_rate, 0.9);
    //auto solver = create_AdadeltaSolver(context::get_current_context(), learning_rate, weight_decay, 1e-08);
    //auto solver = create_AdamSolver(context::get_current_context(), 1e-4, 0.9, 0.999, 1e-08);
    solver.set_parameters(ParameterScope::get_parameters());

    auto eng = std::make_shared<Leela>(&play_model, "");

    float avg_loss = -1;

    go_window my_window;

    std::vector<MoveData> training_data;
    int accum_rounds = 0;

    play_model.load_weights(init_weight_path);
    
    for (int n=0; n < game_rounds; n++) {

        auto sgffile = format_str("game_%d.sgf", n);

        if (accum_rounds++ > clean_per_rounds) {
            accum_rounds = 1;
            training_data.clear();
            play_model.load_weights(weight_path);
            eng->clear_cache();
        }
        

        Training::clear_training();
        int result = eng->selfplay(playouts, sgffile, [&](int move, int board[]) {

            my_window.update(move, board);
        });

        // savinng
        for (const auto& d : Training::m_data) {
            training_data.push_back({d.planes, d.probabilities, d.to_move == FastBoard::BLACK ? result : -result});
        }

        std::random_shuffle(training_data.begin(), training_data.end());

        int rounds = 0;
        int data_index = 0;
        int N = training_data.size();

        std::cout << "total samples = " << N << std::endl;

        if (N < batch_size)
            continue;

        while (rounds < train_rounds) {

            clock_t time=clock();

            bool rewinded = false;
            if (N - data_index < batch_size) {
                data_index = N - batch_size;
                rewinded = true;
            }

            auto this_loss = train_model.train_batch(training_data.begin()+data_index, training_data.begin()+data_index+batch_size, solver, weight_decay);
            data_index += batch_size;

            if(avg_loss == -1) avg_loss = this_loss;
            avg_loss = avg_loss*.95 + this_loss*.05;
            seen += batch_size;
            batchs++;

            if(batchs%10 == 0)
            printf("%ld, (lr=%f) %.3f: %f, %f avg, %lf seconds\n", batchs, learning_rate, (float)seen/N, this_loss, avg_loss, sec(clock()-time));

            if(rewinded) {
                rounds++;
                data_index = 0;
                train_model.save_weights(weight_path, seen, batchs);
            }
        }
    }

    my_window.wait_until_closed();
}



static std::string opt_weights = "../../data/z.weights";
static std::string opt_init_weight;
static int opt_batch_size = 0;
static float opt_learning_rate = 0;
static int opt_games = 100;
static int opt_trains = 30;
static int opt_playouts = 1600;
static int opt_clean_per_rounds = 100;

void parse_commandline( int argc, char **argv ) {
    for (int i = 1; i < argc; i++) { 
        string opt = std::string(argv[i]);
        if (opt.find("--") != 0)
            opt = "";

        if (opt == "--w" || opt == "--weights") {
            opt_weights = argv[++i];
        } else if (opt == "--init") {
            opt_init_weight = argv[++i];
        } else if (opt == "--batch-size" || opt == "--b") {
            opt_batch_size = atoi(argv[++i]);
        } else if (opt == "--lr" || opt == "--learning-rate") {
            opt_learning_rate = atof(argv[++i]);
        } else if (opt == "--games") {
            opt_games = atoi(argv[++i]);
        } else if (opt == "--trains") {
            opt_trains = atoi(argv[++i]);
        } else if (opt == "--p") {
            opt_playouts = atoi(argv[++i]);
        } else if (opt == "--group") {
            opt_clean_per_rounds = atoi(argv[++i]);
        }
    }
}

int main(int argc, char **argv) {
    parse_commandline(argc, argv);
    if (opt_init_weight.empty())
        opt_init_weight = opt_weights;
    train(opt_weights, opt_init_weight, opt_batch_size, opt_learning_rate, opt_games, opt_trains, opt_playouts, opt_clean_per_rounds);
    return 0;
}

