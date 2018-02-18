#pragma once
#include "../AQ/model/dnn/layers.h"
#include "../AQ/model/dnn/core.h"

using namespace dlib;


constexpr int board_size = 19;
constexpr int board_count = board_size*board_size;
constexpr int board_moves = board_count + 1;


template <int N, long kernel, typename SUBNET> using bn_conv2d = affine<con<N,kernel,kernel,1,1,SUBNET>>;

template <int N, typename SUBNET> 
using block  = bn_conv2d<N,3,relu<bn_conv2d<N,3,SUBNET>>>;

template <template <int,typename> class block, int N, typename SUBNET>
using residual = add_prev1<block<N,tag1<SUBNET>>>;

template <int classes, typename SUBNET>
using policy_head = softmax<fc<classes, relu<bn_conv2d<2, 1, SUBNET>>>>;

template <typename SUBNET>
using value_head = htan<fc<1, fc<256, relu<bn_conv2d<1, 1, SUBNET>>>>>;


namespace zero {

constexpr int RESIDUAL_FILTERS = 256;
constexpr int RESIDUAL_BLOCKS = 19;

template <typename SUBNET> 
using ares  = relu<residual<block, RESIDUAL_FILTERS, SUBNET>>;

using net_type = 
                            value_head<
                            skip1<
                            policy_head<board_moves,
                            tag1<
                            repeat<RESIDUAL_BLOCKS, ares,
                            relu<bn_conv2d<RESIDUAL_FILTERS,3,
                            input<matrix<unsigned char>>
                            >>>>>>>;

}

using zero_net_type = zero::net_type;

// leela model
namespace leela {
    
template <typename SUBNET> 
using ares  = relu<residual<block, 128, SUBNET>>;   
    
using net_type = 
                            value_head<
                            skip1<
                            policy_head<board_moves,
                            tag1<
                            repeat<6, ares,
                            relu<bn_conv2d<128,3,
                            input<matrix<unsigned char>>
                            >>>>>>>;
}

using leela_net_type = leela::net_type;

  
template <typename SUBNET> 
using convs  = relu<bn_conv2d<256,3,SUBNET>>;
                                
using dark_net_type =       fc_no_bias<1, // fake value head
                            softmax<
                            fc_no_bias<board_moves,
                            con_bias<1,1,1,1,1,
                            repeat<12, convs,
                            convs<
                            input<matrix<unsigned char>>
                            >>>>>>;

template<typename NET>
inline const tensor& forward(NET& net, const tensor& input, double temperature, const tensor** value_out) {

    if (temperature)
        layer<7>(net).layer_details().set_temprature(temperature);

    if (value_out == nullptr) {
        return layer<7>(net).forward(input);
    }

    *value_out = &net.forward(input);
    return layer<7>(net).get_output();
}

template<>
inline const tensor& forward(dark_net_type& net, const tensor& input, double temperature, const tensor** value_out) {

    if (temperature)
        layer<1>(net).layer_details().set_temprature(temperature);

    if (value_out == nullptr) {
        return layer<1>(net).forward(input);
    }

    *value_out = &net.forward(input);
    return layer<1>(net).get_output();
}



class zero_model {
public:
    static constexpr int num_planes = 18;

    using prediction = std::vector<float>;

    zero_model();

    bool load_weights(const std::string& path);

    template<typename FEATURE>
    std::pair<prediction, float> predict(const FEATURE& ft, double temperature = 1)  {

        cached_input.set_size(1, type() == 0 ? 1 : num_planes, board_size, board_size);

        auto dst = cached_input.host_write_only();
        if (cached_input.k() == 1) {
            for (int i=0; i<board_count; i++) {
                if (ft[0][i]) *dst = 1;
                else if (ft[8][i]) *dst = -1;
                else *dst = 0;
                dst++;
            }
        } else {
            for (int c=0; c< num_planes; c++) {
                for (int i=0; i<board_count; i++) {
                    *(dst++) = (float)ft[c][i];
                }
            }
        }

        const tensor* v_tensor = nullptr;
        auto& out_tensor = forward(cached_input, temperature, &v_tensor);
        auto src = out_tensor.host();
        auto data = v_tensor->host();

        prediction dist(board_moves);
        std::copy(src, src+board_moves, dist.begin());
        float val = data[0];

        return {dist, val};
    }

    int type() const {
        if (zero_weights_loaded)
            return 1;
        else if(leela_weights_loaded)
            return 2;
        else
            return 0;
    }

private:
    const tensor& forward(const tensor& input, double temperature, const tensor** value_out);

    zero_net_type zero_net;
    dark_net_type dark_net;
    leela_net_type leela_net;
    resizable_tensor cached_input;
    bool zero_weights_loaded;
    bool leela_weights_loaded;
};
