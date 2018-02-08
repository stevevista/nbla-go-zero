#pragma once

// DNN module uses template-based network declaration that leads to very long
// type names. Visual Studio will produce Warning C4503 in such cases
#ifdef _MSC_VER
#   pragma warning( disable: 4503 )
#endif

#include "dnn/layers.h"
#include "dnn/core.h"
#include <bitset>
#include <array>

namespace lightmodel {

using namespace dlib;

constexpr int board_size = 19;
constexpr int board_count = board_size*board_size;
constexpr int board_moves = board_count + 1;

constexpr int RESIDUAL_FILTERS = 256;
constexpr int RESIDUAL_BLOCKS = 19;


template <int N, long kernel, typename SUBNET> using bn_conv2d = affine<con<N,kernel,kernel,1,1,SUBNET>>;

template <int N, typename SUBNET> 
using block  = bn_conv2d<N,3,relu<bn_conv2d<N,3,SUBNET>>>;

template <template <int,typename> class block, int N, typename SUBNET>
using residual = add_prev1<block<N,tag1<SUBNET>>>;

template <typename SUBNET> 
using ares  = relu<residual<block, RESIDUAL_FILTERS, SUBNET>>;

template <int classes, typename SUBNET>
using policy_head = softmax<fc<classes, relu<bn_conv2d<2, 1, SUBNET>>>>;

template <typename SUBNET>
using value_head = htan<fc<1, fc<256, relu<bn_conv2d<1, 1, SUBNET>>>>>;

using zero_net_type = 
                                value_head<
                                skip1<
                                policy_head<board_moves,
                                tag1<
                                repeat<RESIDUAL_BLOCKS, ares,
                                relu<bn_conv2d<RESIDUAL_FILTERS,3,
                                input
                                >>>>>>>;

template <typename SUBNET> 
using convs  = relu<bn_conv2d<256,3,SUBNET>>;

using dark_net_type =           softmax<
                                fc_no_bias<board_moves,
                                con_bias<1,1,1,1,1,
                                repeat<12, convs,
                                convs<
                                input
                                >>>>>;


// leela model
namespace leela {

template <typename SUBNET> 
using ares  = relu<residual<block, 128, SUBNET>>;   

using leela_net_type = 
                                value_head<
                                skip1<
                                policy_head<board_moves,
                                tag1<
                                repeat<6, ares,
                                relu<bn_conv2d<128,3,
                                input
                                >>>>>>>;
}

using leela::leela_net_type;


bool load_zero_weights(zero_net_type& net, const std::string& path);
bool load_dark_weights(dark_net_type& net, const std::string& path);
bool load_leela_weights(leela_net_type& net, const std::string& path);

class zero_model {
public:
    struct netres {
        int index;
        double score;
    };

    static constexpr int num_planes = 18;

    using plane = std::bitset<board_count>;
    using feature = std::array<plane, num_planes>;
    using prediction = std::vector<float>;
    using prediction_ex = std::pair<prediction, float>;

    zero_model();

    bool load_weights(const std::string& path);
    void set_batch_size(size_t size);

    prediction predict_policy(const feature& ft, double temperature = 1, bool suppress_invalid = false, bool darknet_backend = false);
    float predict_value(const feature& ft);
    prediction_ex predict(const feature& ft, double temperature = 1, bool suppress_invalid = false, bool darknet_backend = false);

    std::vector<prediction> predict_policy(const std::vector<feature>& features, double temperature = 1, 
                                    bool suppress_invalid = false, 
                                    bool darknet_backend = false);

    std::vector<float> predict_value(const std::vector<feature>& features);

    std::vector<prediction_ex> predict(
                                    const std::vector<feature>& features, double temperature = 1, 
                                    bool suppress_invalid = false, bool darknet_backend = false);

    std::vector<netres> predict_top(const feature& input, int top_n=1, double temperature = 1, bool darknet_backend = false);

private:

    void predict_policies(
            std::vector<feature>::const_iterator begin, 
            std::vector<feature>::const_iterator end, 
            std::vector<prediction>::iterator it, 
            double temperature, 
            bool darknet_backend);

    void predict_values(
            std::vector<feature>::const_iterator begin, 
            std::vector<feature>::const_iterator end,
            std::vector<float>::iterator it);

    void predict(
            std::vector<feature>::const_iterator begin, 
            std::vector<feature>::const_iterator end,
            std::vector<prediction_ex>::iterator it, 
            double temperature, 
            bool darknet_backend);

    const tensor& forward(const tensor& input, double temperature, bool policy_only);

    template<typename ITER>
    const tensor& features_to_tensor(
                ITER begin, 
                ITER end, 
                bool darknet_backend) {

        const int batch_size = std::distance(begin, end);
        cached_input.set_size(batch_size, darknet_backend ? 1 : num_planes, board_size, board_size);

        auto dst = cached_input.host_write_only();
        for (int n=0; n<batch_size; n++, begin++) {
                
            if (cached_input.k() == 1) {
                for (int i=0; i<board_count; i++) {
                    if ((*begin)[0][i]) *dst = 1;
                    else if ((*begin)[8][i]) *dst = -1;
                    else *dst = 0;
                    dst++;
                }
            } else {
                for (int c=0; c< num_planes; c++) {
                    for (int i=0; i<board_count; i++) {
                        *(dst++) = (float)(*begin)[c][i];
                    }
                }
            }
        }

        return cached_input;
    }


    void suppress_policy(prediction& prob, const feature& ft);

    zero_net_type zero_net;
    dark_net_type dark_net;
    leela_net_type leela_net;
    size_t max_batch_size;
    tensor cached_input;
    bool zero_weights_loaded;
    bool leela_weights_loaded;
};


}


