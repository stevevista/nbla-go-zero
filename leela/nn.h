#pragma once
#include "nn/dnn/layers.h"
#include "nn/dnn/core.h"
#include "size_info.h"

using namespace dlib;

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

template <typename SUBNET> 
using ares  = relu<residual<block, RESIDUAL_FILTERS, SUBNET>>;

using net_type = 
                            value_head<
                            skip1<
                            policy_head<board_moves,
                            tag1<
                            repeat<RESIDUAL_BLOCKS, ares,
                            relu<bn_conv2d<RESIDUAL_FILTERS,3,
                            input
                            >>>>>>>;

}

using zero_net_type = zero::net_type;



class zero_model {
public:
    using prediction = std::vector<float>;

    zero_model();

    bool load_weights(const std::string& path);

    std::pair<prediction, float> forward_net(double temperature = 1)  {

        if (temperature != 1)
            layer<7>(zero_net).layer_details().set_temprature(temperature);

        auto& value_out = zero_net.forward(cached_input);
        auto& out_tensor = layer<7>(zero_net).get_output();

        auto src = out_tensor.host();
        auto data = value_out.host();

        prediction dist;
        std::copy(src, src+ zero::board_moves, std::back_inserter(dist)); 
        float val = data[0];

        return {dist, val};
    }

    template<typename FEATURE>
    std::pair<prediction, float> predict(const FEATURE& ft, double temperature = 1)  {

        using namespace zero;
        
        auto dst = input_buffer();
        for (int c=0; c< input_channels; c++) {
            for (int i=0; i<board_count; i++) {
                *(dst++) = (float)ft[c][i];
            }
        }

        return forward_net(temperature);
    }

    float* input_buffer() { 
        using namespace zero;
        cached_input.set_size(1, input_channels, board_size, board_size);
        return cached_input.host_write_only(); 
    }

private:
    zero_net_type zero_net;
    resizable_tensor cached_input;
};
