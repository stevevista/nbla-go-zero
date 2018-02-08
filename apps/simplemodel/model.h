// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_
#define DLIB_DNn_

// DNN module uses template-based network declaration that leads to very long
// type names. Visual Studio will produce Warning C4503 in such cases
#ifdef _MSC_VER
#   pragma warning( disable: 4503 )
#endif

#include "dnn/layers.h"
#include "dnn/core.h"
#include <bitset>
#include <array>

namespace dlib {

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
                                policy_head<board_moves,
                                skip1<
                                value_head<
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


bool load_zero_weights(zero_net_type& net, const std::string& path);
bool load_dark_weights(dark_net_type& net, const std::string& path);

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

    const tensor& forward(const tensor& input, double temperature);

    const tensor& features_to_tensor(std::vector<feature>::const_iterator begin, std::vector<feature>::const_iterator end, bool darknet_backend);

    void suppress_policy(prediction& prob, const feature& ft);

    zero_net_type zero_net;
    dark_net_type dark_net;
    size_t max_batch_size;
    tensor cached_input;
    bool zero_weights_loaded;
};


}



#endif // DLIB_DNn_

