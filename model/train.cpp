
#include <nblapp/solver.hpp>
#include <nblapp/parameter.hpp>
#include <model/utils.hpp>
#include <model/zero_model.hpp>

#include <fstream>
#include <cassert>


using std::make_shared;
using namespace nblapp;
using namespace napp;



float sec(clock_t clocks)
{
    return (float)clocks/CLOCKS_PER_SEC;
}

void train(
    const std::string& data_path, 
    const std::string& weight_path, 
    int batch_size, 
    float learning_rate) {

    GoBoard::init_board();

    float weight_decay = 0.0005;

    GameArchive m;
    auto N = m.load(data_path, true);
    if (N == 0)
        throw std::runtime_error("cannot load training data");

    const int boardsize = m.boardsize;
    std::cout << "training data, boardsize = " << boardsize << std::endl;
    std::cout << "total samples = " << N << std::endl;

    ZeroTrainModel train_model(batch_size, 0.1, true);
    train_model.load_weights(weight_path);

    batch_size = train_model.get_batch_size();
    if (learning_rate == 0) {
        learning_rate = train_model.get_learning_rate();
    }
    int seen = train_model.get_seen();
    int batchs = train_model.get_batchs();
    int rounds = train_model.get_rounds();

 
    auto save_weights = [&](const std::string& path, int rounds) {

        train_model.save_weights(path, seen, batchs, rounds, learning_rate);
        std::cout << "weights saved!" << std::endl;
    };

    std::cout << "batch_size = " << batch_size << std::endl;
    std::cout << "learning_rate = " << learning_rate << std::endl;


    auto solver = MomentumSolver(learning_rate, 0.9);
    //auto solver = SgdSolver(learning_rate);

    //auto solver = create_AdadeltaSolver(context::get_current_context(), learning_rate, weight_decay, 1e-08);
    //auto solver = create_AdamSolver(context::get_current_context(), 1e-4, 0.9, 0.999, 1e-08);

    solver.set_parameters(ParameterScope::get_parameters());

    //save_weights(weight_path, 0);
    //return;

    m.data_index = seen % N;

    float avg_loss = -1;

    while (rounds < 100) {

        clock_t time=clock();

        bool rewinded;
        auto batch = m.next_batch(batch_size, rewinded);

        auto this_loss = train_model.train_batch(batch.begin(), batch.end(), solver, weight_decay);

        if (rewinded)
            rounds++;

        if(avg_loss == -1) avg_loss = this_loss;
        avg_loss = avg_loss*.95 + this_loss*.05;
        seen += batch_size;
        batchs++;

        if(batchs % 15 == 0)
        printf("%ld, (lr=%f) %.3f: %f, %f avg, %lf seconds\n", batchs, learning_rate, (float)seen/N, this_loss, avg_loss, sec(clock()-time));

        if(batchs % 1000 == 0) {
            save_weights(weight_path, rounds);

            //learning_rate -= learning_rate*0.1;
            // solver.set_learning_rate(learning_rate);
        }

        if(batchs%10000 == 0) {
            save_weights(weight_path + format_str(".%d", batchs), rounds);
        }
    }

}


static std::string opt_weights = "../../data/leela.weights";
static std::string opt_input = "../../data/train.data";
static int opt_batch_size = 0;
static float opt_learning_rate = 0;

void parse_commandline( int argc, char **argv ) {
    for (int i = 1; i < argc; i++) { 
        string opt = argv[i];
 
        if (opt == "--w" || opt == "--weights") {
            opt_weights = argv[++i];
        } else if (opt == "--input") {
            opt_input = argv[++i];
        } else if (opt == "--batch-size" || opt == "--b") {
            opt_batch_size = atoi(argv[++i]);
        } else if (opt == "--lr" || opt == "--learning-rate") {
            opt_learning_rate = atof(argv[++i]);
        }
    }
}

int main(int argc, char **argv) {
    parse_commandline(argc, argv);
    train(opt_input, opt_weights, opt_batch_size, opt_learning_rate);
    return 0;
}

