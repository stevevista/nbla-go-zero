#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/computation_graph/function.hpp>
#include <nbla/function/convolution.hpp>
#include <nbla/function/batch_normalization.hpp>
#include <nbla/function/affine.hpp>
#include <nbla/function/relu.hpp>
#include <nbla/function/tanh.hpp>
#include <nbla/function/softmax.hpp>
#include <nbla/function/add2.hpp>
#include <nbla/function/sink.hpp>
#include <nbla/function/mul_scalar.hpp>
#include <nbla/init.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace nbla {

using std::make_shared;
using std::vector;
using std::ifstream;
using std::string;


CgVariablePtr conv2d(const Context &ctx, CgVariablePtr x, CgVariablePtr w) {

    auto shape_w = w->variable()->shape();
    NBLA_CHECK(shape_w.size() == 4, error_code::value,
             "Shape of conv2d weights must be 4 dimention tensor (filters, input, h, w). "
             "Shape size is : %d.",
             shape_w.size());

    int kh = shape_w[2];
    int kw = shape_w[3];
    vector<int> pad = {kh/2, kw/2};
    
    auto op = make_shared<CgFunction>(create_Convolution(ctx, 1, pad, /*stride*/{1, 1}, /*dilation*/{1, 1}, /*group*/1));
    return connect(op, {x, w});
}

CgVariablePtr bn_conv2d(const Context &ctx, CgVariablePtr x, CgVariablePtr w, CgVariablePtr b, CgVariablePtr g) {

    auto conv = conv2d(ctx, x, w);
    auto op = make_shared<CgFunction>(create_BatchNormalization(ctx, 1));
    return connect(op, {conv, b, g});
}

CgVariablePtr fc(const Context &ctx, CgVariablePtr x, CgVariablePtr w, CgVariablePtr b) {
    auto op = make_shared<CgFunction>(create_Affine(ctx, 1));
    return connect(op, {x, w, b});
}

CgVariablePtr relu(const Context &ctx, CgVariablePtr x) {
    auto op = make_shared<CgFunction>(create_ReLU(ctx));
    return connect(op, {x});
}

CgVariablePtr add(const Context &ctx, CgVariablePtr a, CgVariablePtr b) {
    auto op = make_shared<CgFunction>(create_Add2(ctx));
    return connect(op, {a, b});
}

CgVariablePtr tanh(const Context &ctx, CgVariablePtr x) {
    auto op = make_shared<CgFunction>(create_Tanh(ctx));
    return connect(op, {x});
}

CgVariablePtr softmax(const Context &ctx, CgVariablePtr x) {

    auto op = make_shared<CgFunction>(create_Softmax(ctx, 1));
    return connect(op, {x});
}

CgVariablePtr mul(const Context &ctx, CgVariablePtr x, CgVariablePtr temp) {
    auto op = make_shared<CgFunction>(create_MulScalar(ctx));
    return connect(op, {x, temp});
}


vector<CgVariablePtr> create_model(const Context &ctx, CgVariablePtr x, const vector<CgVariablePtr>& weights) {

    auto shape_x = x->variable()->shape();
    NBLA_CHECK(shape_x.size() == 4, error_code::value,
             "Shape of input must be 4 dimention tensor (NCHW). "
             "Shape size is : %d.",
             shape_x.size());

    int w_index = 0;

    auto get_bnconv_weights = [&weights, &w_index]() ->vector<CgVariablePtr> {

        NBLA_CHECK(weights.size() - w_index >= 3, error_code::value,
             "Not enough weights for bn conv. "
             "Left is : %d. cur index is %d.",
             weights.size() - w_index, w_index);

        auto w = weights[w_index++];
        auto b = weights[w_index++];
        auto g = weights[w_index++];
        return { w, b, g };
    };

    auto get_fc_weights = [&weights, &w_index]() ->vector<CgVariablePtr> {

        NBLA_CHECK(weights.size() - w_index >= 2, error_code::value,
             "Not enough weights for fc. "
             "Left is : %d. cur index is %d.",
             weights.size() - w_index, w_index);

        auto w = weights[w_index++];
        auto b = weights[w_index++];
        return { w, b };
    };
    
    // First layer is convolution with batch_normalization, relu 
    // get weights
    auto W = get_bnconv_weights();
    auto h = relu(ctx, bn_conv2d(ctx, x, W[0], W[1], W[2]));

    while (true) {
        // check how many RESIDUAL BLOCKS there
        // jugde by out filters
        // filters == 2, policy head
        // otherwise, residual blocks 
        W = get_bnconv_weights();
        int filters = W[0]->variable()->shape()[0];

        if (filters == 2) {
            break;
        }

        auto orig = h;
        h = relu(ctx, bn_conv2d(ctx, h, W[0], W[1], W[2]));

        W = get_bnconv_weights();
        h = bn_conv2d(ctx, h, W[0], W[1], W[2]);
        h = relu(ctx, add(ctx, h, orig));
    }

    // policy head 
    auto ph = relu(ctx, bn_conv2d(ctx, h, W[0], W[1], W[2]));
    W = get_fc_weights();
    ph = fc(ctx, ph, W[0], W[1]);

    auto temperature = make_shared<CgVariable>();
    ph = mul(ctx, ph, temperature);
    ph = softmax(ctx, ph);

    // value head 
    W = get_bnconv_weights();
    auto vh = relu(ctx, bn_conv2d(ctx, h, W[0], W[1], W[2]));
    W = get_fc_weights();
    vh = relu(ctx, fc(ctx, vh, W[0], W[1]));  // fc1 
    W = get_fc_weights();
    vh = fc(ctx, vh, W[0], W[1]);  // fc2 
    vh = tanh(ctx, vh);

    // linkup, so we can forward graph 
    auto lnk_op = make_shared<CgFunction>(create_Sink(ctx));
    vh = connect(lnk_op, {vh, ph});

    return { vh, ph, temperature };
}


vector<CgVariablePtr> load_leela_weights(const std::string& path) {

    ifstream ifs(path);
    if (ifs.fail())
        return {};
    
    int channles = 0;

    int linecount = 0;
    string line;
    while (getline(ifs, line)) {

        std::istringstream iss(line);
        if (linecount == 0) {
            int version;
            iss >> version;
            if (version != 1)
                throw std::runtime_error("unknown leela weights version");
        } 

        // Third line of parameters are the convolution layer biases,
        // so this tells us the amount of channels in the residual layers.
        // (Provided they're all equally large - that's not actually required!)
        if (linecount == 2) {
            channles = std::distance(std::istream_iterator<std::string>(iss),
                                       std::istream_iterator<std::string>());
            std::cerr << channles << " channels..." << std::endl;
        }

        linecount++;
    }

    // 1 format id, 1 input layer (4 x weights), 14 ending weights,
    // the rest are residuals, every residual has 8 x weight lines
    auto residual_blocks = linecount - (1 + 4 + 14);
    if (residual_blocks % 8 != 0) {
        throw std::runtime_error("Inconsistent number of weights in the file.");
    }

    residual_blocks /= 8;
    // std::cerr << residual_blocks << " blocks" << std::endl;

    // Re-read file and process
    ifs.clear();
    ifs.seekg(0, std::ios::beg);

    // Get the file format id out of the way
    std::getline(ifs, line);

    vector<CgVariablePtr> params;

    std::vector<float> bias;
    std::vector<float> rm;

    auto convert_to_gemma_beta = [&params, &bias, &rm](const vector<float>& rv) {

        constexpr float eps = 1e-5;

        int channels = (int)rv.size();

        auto b = make_shared<CgVariable>(Shape_t{1, channels, 1, 1});
        auto g = make_shared<CgVariable>(Shape_t{1, channels, 1, 1});
        auto beta = b->variable()->cast_data_and_get_pointer<float>(Context());
        auto gamma = g->variable()->cast_data_and_get_pointer<float>(Context());

        for (int i=0; i<channels; i++) {
            float scale = 1.0f/sqrt(rv[i]+eps);
            float fixed_beta = (bias[i] - rm[i]) * scale;
            gamma[i] = scale;
            beta[i] = fixed_beta;
        }

        params.emplace_back(b);
        params.emplace_back(g);
    };

    auto add_fc_W = [&params](std::vector<float>& weights, int inputs, int outputs) {

        auto w = make_shared<CgVariable>(Shape_t{inputs, outputs});
        auto dst = w->variable()->cast_data_and_get_pointer<float>(Context());

        // transpose from [out, in] -> [in, out]
        for(int x = 0; x < outputs; ++x) {
            for(int y = 0; y < inputs; ++y){
                dst[y*outputs + x] = weights[x*inputs + y];
            }
        }

        params.emplace_back(w);
    };

    const auto plain_conv_layers = 1 + (residual_blocks * 2);
    auto plain_conv_wts = plain_conv_layers * 4;
    linecount = 0;
    while (std::getline(ifs, line)) {
        std::vector<float> weights;
        float weight;
        std::istringstream iss(line);
        while (iss >> weight) {
            weights.emplace_back(weight);
        }

        if (linecount < plain_conv_wts) {
            if (linecount % 4 == 0) {
                // conv/W
                int insize = weights.size() / (channles*3*3);
                auto w = make_shared<CgVariable>(Shape_t{channles, insize, 3, 3});
                std::copy(weights.begin(), weights.end(), w->variable()->cast_data_and_get_pointer<float>(Context()));
                params.emplace_back(w);
            } else if (linecount % 4 == 1) {
                bias = weights;
            } else if (linecount % 4 == 2) {
                // running mean
                rm = weights;
            } else if (linecount % 4 == 3) {
                // running var
                convert_to_gemma_beta(weights);
            } 
        } else if (linecount == plain_conv_wts) {
            // policy_head/conv/W 
            auto w = make_shared<CgVariable>(Shape_t{2, channles, 1, 1});
            std::copy(weights.begin(), weights.end(), w->variable()->cast_data_and_get_pointer<float>(Context()));
            params.emplace_back(w);

        } else if (linecount == plain_conv_wts + 1) {
            bias = weights;
            
        } else if (linecount == plain_conv_wts + 2) {
            rm = weights;

        } else if (linecount == plain_conv_wts + 3) {

            convert_to_gemma_beta(weights);
        } else if (linecount == plain_conv_wts + 4) {
            // fc/W
            add_fc_W(weights, 2*361, 362);

        } else if (linecount == plain_conv_wts + 5) {
            // fc/b
            auto w = make_shared<CgVariable>(Shape_t{362});
            std::copy(weights.begin(), weights.end(), w->variable()->cast_data_and_get_pointer<float>(Context()));
            params.emplace_back(w);
        } else if (linecount == plain_conv_wts + 6) {
            // value_head/conv/W
            auto w = make_shared<CgVariable>(Shape_t{1, channles, 1, 1});
            std::copy(weights.begin(), weights.end(), w->variable()->cast_data_and_get_pointer<float>(Context()));
            params.emplace_back(w);
            
        } else if (linecount == plain_conv_wts + 7) {
            bias = weights;
           
        } else if (linecount == plain_conv_wts + 8) {
            rm = weights;

        } else if (linecount == plain_conv_wts + 9) {
            convert_to_gemma_beta(weights);

        } else if (linecount == plain_conv_wts + 10) {
            add_fc_W(weights, 361, 256);

        } else if (linecount == plain_conv_wts + 11) {
            auto w = make_shared<CgVariable>(Shape_t{256});
            std::copy(weights.begin(), weights.end(), w->variable()->cast_data_and_get_pointer<float>(Context()));
            params.emplace_back(w);

        } else if (linecount == plain_conv_wts + 12) {
            add_fc_W(weights, 256, 1);

        } else if (linecount == plain_conv_wts + 13) {
            auto w = make_shared<CgVariable>(Shape_t{1});
            std::copy(weights.begin(), weights.end(), w->variable()->cast_data_and_get_pointer<float>(Context()));
            params.emplace_back(w);
        }

        linecount++;
    }

    return params;
}


}


void test_lite() {

    using namespace nbla;
    init_cpu();

    auto x = make_shared<CgVariable>(Shape_t{1,18,19,19});
 
    auto ws = load_leela_weights("../../data/weights.txt");
    auto outs = create_model(Context(), x, ws);
    auto v = outs[0];
    auto p = outs[1];
    auto temp = outs[2];
    temp->variable()->fill(1);

    auto x_data = x->variable()->cast_data_and_get_pointer<float>(Context());
    std::fill(x_data, x_data+18*19*19, 0);
    std::fill(x_data+16*19*19, x_data+17*19*19, 1);

    // x->variable()->zero();
    v->forward();

    std::cout << p->variable()->size() << std::endl;

    auto dist = p->variable()->get_data_pointer<float>(Context());
    for (int i=0; i<19; i++) {
        for (int j=0; j<19; j++) {
            std::cout << std::setw(3) << (int)((*(dist++))*1000) << " ";
        }
        std::cout << std::endl;
    }
}
