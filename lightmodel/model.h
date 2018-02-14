#pragma once

// DNN module uses template-based network declaration that leads to very long
// type names. Visual Studio will produce Warning C4503 in such cases
#ifdef _MSC_VER
#   pragma warning( disable: 4503 )
#endif

#include "dnn/layers.h"
#include "dnn/core.h"
#include <bitset>

// For windows support
#if defined(_MSC_VER) && !defined(__CUDACC__)
#ifdef lightmodel_EXPORTS
#define LMODEL_API __declspec(dllexport)
#else
#define LMODEL_API __declspec(dllimport)
#endif
#else
#define LMODEL_API
#endif


namespace lightmodel {

using namespace dlib;

using TFEATURE = std::array<std::bitset<361>, 18>;

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

using dark_net_type =           fc_no_bias<1, // fake value head
                                softmax<
                                fc_no_bias<board_moves,
                                con_bias<1,1,1,1,1,
                                repeat<12, convs,
                                convs<
                                input
                                >>>>>>;


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
    struct netres {
        int index;
        double score;
    };

    static constexpr int num_planes = 18;

    using prediction = std::vector<float>;
    using prediction_ex = std::pair<prediction, float>;

    LMODEL_API zero_model();

    
prediction_ex predict2(
    const TFEATURE& features, double temperature, 
    bool suppress_invalid = false);

    LMODEL_API bool load_weights(const std::string& path);
    LMODEL_API void set_batch_size(size_t size);

    template<typename FEATURE>
    prediction predict_policy(const FEATURE& ft, double temperature = 1, bool suppress_invalid = false) {

        std::vector<prediction> out(1);

        predict_batch_policies(features_to_tensor(&ft, &ft+1), out.begin(), temperature);
        if (suppress_invalid)
            suppress_policy(out[0], ft);
        return out[0];
    }

    template<typename FEATURE>
    prediction_ex predict(const FEATURE& ft, double temperature = 1, bool suppress_invalid = false)  {

        std::vector<prediction_ex> out(1);
        std::cout << "zero w " << this->zero_weights_loaded << std::endl;
        predict_batch(features_to_tensor(&ft, &ft+1), out.begin(), temperature);
        if (suppress_invalid)
            suppress_policy(out[0].first, ft);

        return out[0];
    }

    template<typename FEATURE>
    float predict_value(const FEATURE& ft) {

        std::vector<float> out(1);
        predict_batch_values(features_to_tensor(&ft, &ft+1), out.begin());
        return out[0];
    }


    template<typename FEATURE>
    std::vector<prediction> predict_policy(const std::vector<FEATURE>& features, double temperature = 1, 
                                    bool suppress_invalid = false)  {

        std::vector<prediction> out(features.size());

        auto in_it = features.begin();
        auto out_it = out.begin();
        auto count = features.size();
        while (count > 0) {
            auto batch_size = std::min(count, max_batch_size);
            predict_batch_policies(features_to_tensor(in_it, in_it+batch_size), out_it, temperature);

            if (suppress_invalid) {
                for (int i=0; i<batch_size; i++) {
                    suppress_policy(*(out_it+i), *(in_it+i));
                }
            }

            in_it += batch_size;
            out_it += batch_size;
            count -= batch_size;
        }

        return out;
    }

    template<typename FEATURE>
    std::vector<prediction_ex> predict(
                                    const std::vector<FEATURE>& features, double temperature = 1, 
                                    bool suppress_invalid = false) {

        std::vector<prediction_ex> out(features.size());

        auto in_it = features.begin();
        auto out_it = out.begin();
        auto count = features.size();
        while (count > 0) {
            auto batch_size = std::min(count, max_batch_size);
            predict_batch(features_to_tensor(in_it, in_it+batch_size), out_it, temperature);

            if (suppress_invalid) {
                for (auto i=0; i<batch_size; i++) {
                    suppress_policy((out_it+i)->first, *(in_it+i));
                }
            }

            in_it += batch_size;
            out_it += batch_size;
            count -= batch_size;
        }

        return out;
    }

    template<typename FEATURE>
    std::vector<float> predict_value(const std::vector<FEATURE>& features) {

        std::vector<float> out(features.size());

        auto in_it = features.begin();
        auto out_it = out.begin();
        auto count = features.size();
        while (count > 0) {
            auto batch_size = std::min(count, max_batch_size);
            predict_batch_values(features_to_tensor(in_it, in_it+batch_size), out_it);
            in_it += batch_size;
            out_it += batch_size;
            count -= batch_size;
        }

        return out;
    }

    template<typename FEATURE>
    std::vector<netres> predict_top(const FEATURE& ft, int top_n=1, double temperature = 1) {

        auto probs = predict_policy(ft, temperature, true);

        std::vector<netres> index(top_n);

        for(auto& e : index) e.index = -1;
        for(int i = 0; i < probs.size(); ++i) {
            int curr = i;
            for(auto& e : index) {
                if((e.index < 0) || probs[curr] > probs[e.index]) {
                    int swap = curr;
                    curr = e.index;
                    e.index = swap;
                    e.score = probs[e.index];
                }
            }
        }

        return index;
    }

private:
    LMODEL_API void predict_batch(const tensor& input, std::vector<prediction_ex>::iterator it, double temperature);
    LMODEL_API void predict_batch_policies(const tensor& input, std::vector<prediction>::iterator it, double temperature);
    LMODEL_API void predict_batch_values(const tensor& input, std::vector<float>::iterator it);

    LMODEL_API const tensor& forward(const tensor& input, double temperature, const tensor** value_out);

    template<typename ITER>
    const tensor& features_to_tensor(
                ITER begin, 
                ITER end) {

        const int batch_size = std::distance(begin, end);
        bool darknet_backend = (!zero_weights_loaded && !leela_weights_loaded);
        cached_input.set_size(batch_size, darknet_backend ? 1 : num_planes, board_size, board_size);

        auto dst = cached_input.host_write_only();
        for (; begin != end; begin++) {
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


    template<typename FEATURE>
    void suppress_policy(prediction& prob, const FEATURE& ft) {

        auto& b0 = ft[0];
        auto& b1 = ft[8];

        for (int i=0; i<board_count; i++)
            if (b0[i] || b1[i])
                prob[i] = 0;
    }

    zero_net_type zero_net;
    dark_net_type dark_net;
    leela_net_type leela_net;
    size_t max_batch_size;
    tensor cached_input;
    bool zero_weights_loaded;
    bool leela_weights_loaded;
};


}


