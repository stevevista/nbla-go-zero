#pragma once
#include "../AQ/model/model.h"

using namespace dlib;
using namespace lightmodel;


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
    using prediction_ex = std::pair<prediction, float>;

    LMODEL_API zero_model();

    LMODEL_API bool load_weights(const std::string& path);

    template<typename FEATURE>
    prediction_ex predict(const FEATURE& ft, double temperature = 1)  {

        auto& input = features_to_tensor(&ft, &ft+1);
        const tensor* v_tensor = nullptr;
        auto& out_tensor = forward(input, temperature, &v_tensor);
        auto src = out_tensor.host();
        auto data = v_tensor->host();

        prediction dist(board_moves);
        std::copy(src, src+board_moves, dist.begin());
        src += board_moves;
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

    zero_net_type zero_net;
    dark_net_type dark_net;
    leela_net_type leela_net;
    resizable_tensor cached_input;
    bool zero_weights_loaded;
    bool leela_weights_loaded;
};
