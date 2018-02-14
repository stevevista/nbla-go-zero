#include "model.h"
#include <iostream>
#include <fstream>

namespace lightmodel {


    static bool restore_string(std::istream& is, std::string& str) {
        
            int sz;
            is.read((char*)&sz, 4);
            if (sz <=0 || sz > 1024)
                return false;
        
            str.resize(sz);
            if (is.read(&str[0], sz).gcount() != sz)
                return false;
        
            if (strlen(&str[0]) != sz)
                return false;
        
            return true;
        }
        
        static bool restore_variable_dim(std::istream& is, std::vector<int>& shape) {
        
            // skip need_grad flag
            char c;
            if (is.read(&c, 1).gcount() != 1)
                return false;
        
          int dim;
          if (is.read((char*)&dim, 4).gcount() != 4)
            return false;
        
          for (int i=0; i<dim; i++) {
            int d;
            if (is.read((char*)&d, 4).gcount() != 4)
              return false;
            shape.push_back(d);
          }
        
          return true;
        }
        
        
static bool end_with(const std::string& s, const std::string& sub) {
    auto pos = s.rfind(sub);
    return pos != std::string::npos && pos == s.size() - sub.size();
}
        
        
bool load_zero_weights(zero_net_type& net, const std::string& path) {
        
    std::ifstream ifs(path, std::ifstream::binary);
    if (ifs.fail())
        return false;
        
            // std::cout << net << std::endl;
        
            // sort params in stack
    constexpr int block_w_offset = 5;
    constexpr int policy_w_offset = block_w_offset + 19*10;
    constexpr int value_w_offset = policy_w_offset + 5 + 2;
        
    std::vector<param_data> params(value_w_offset + 5 + 4); 
        
            while(true) {
                std::string key;
                std::vector<int> shape;
        
                if (!restore_string(ifs, key))
                    break;
        
                int offset = -1;
                int block_sub = 0;
        
                if (key.find("init/") == 0) {
                    offset = 0;
                } 
                else if (key.find("block_") == 0) {
                    // block_7/0/bn/b
                    auto pos = key.find('/');
                    auto idx = std::stoi(key.substr(6, pos - 6));
                    block_sub = std::stoi(key.substr(pos+1, 1));
                    offset = block_w_offset + (idx-1)*10;
                }
                else if (key.find("policy_head/bn/") == 0) {
                    offset = policy_w_offset;
                    
                }
                else if (key == "policy_head/fc/fc/W") {
                    offset = policy_w_offset + 5;
                }
                else if (key == "policy_head/fc/fc/b") {
                    offset = policy_w_offset + 6;
                }
                else if (key.find("value_head/bn/") == 0) {
                    offset = value_w_offset;
                }
                else if (key == "value_head/fc1/fc/W") {
                    offset = value_w_offset + 5;
                }
                else if (key == "value_head/fc1/fc/b") {
                    offset = value_w_offset + 6;
                }
                else if (key == "value_head/fc2/fc/W") {
                    offset = value_w_offset + 7;
                }
                else if (key == "value_head/fc2/fc/b") {
                    offset = value_w_offset + 8;
                }
        
                if (end_with(key, "conv/W")) {
                    offset += 0 + block_sub*5;
                }
                else if (end_with(key, "bn/g")) {
                    offset += 1 + block_sub*5;
                }
                else if (end_with(key, "bn/b")) {
                    offset += 2 + block_sub*5;
                }
                else if (end_with(key, "bn/m")) {
                    offset += 3 + block_sub*5;
                }
                else if (end_with(key, "bn/v")) {
                    offset += 4 + block_sub*5;
                }
        
            
                if (!restore_variable_dim(ifs, shape))
                    return false;
        
                int count = 1;
                for (auto d : shape) count*=d;
                std::vector<float> data(count);
                if (ifs.read((char*)(&data[0]), count*sizeof(float)).gcount() != count*sizeof(float))
                    return false;
        
                if (offset >=0)
                    params[offset] = {shape, data};
            }
        
    net.consume_params(params.begin());
    return true;
}




bool load_dark_weights(dark_net_type& net, const std::string& path) {
    
        std::ifstream ifs(path, std::ifstream::binary);
        if (ifs.fail())
            return false;
    
        
        std::vector<param_data> go_params; 
    
        int major;
            int minor;
            int revision;
            ifs.read((char*)&major, 4);
            ifs.read((char*)&minor, 4);
            ifs.read((char*)&revision, 4);
            
        if ((major*10 + minor) > 20){
            return false;
        }
            
            if ((major*10 + minor) >= 2){
                size_t seen;
                ifs.read((char*)&seen, sizeof(size_t));
            } else {
                int seen;
                ifs.read((char*)&seen, 4);
            }
    
        int k = 1;
        for (int i=0; i<13; ++i) {
            std::vector<float> b(256);
            std::vector<float> g(256);
            std::vector<float> m(256);
            std::vector<float> v(256);
            std::vector<float> w(256*k*3*3);
            
            ifs.read((char*)&b[0], sizeof(float)*b.size());
            ifs.read((char*)&g[0], sizeof(float)*g.size());
            ifs.read((char*)&m[0], sizeof(float)*m.size());
            ifs.read((char*)&v[0], sizeof(float)*v.size());
            ifs.read((char*)&w[0], sizeof(float)*w.size());
    
            go_params.push_back({std::vector<int>{256, k, 3, 3}, w});
            go_params.push_back({std::vector<int>{1, 256, 1, 1}, g});
            go_params.push_back({std::vector<int>{1, 256, 1, 1}, b});
            go_params.push_back({std::vector<int>{1, 256, 1, 1}, m});
            go_params.push_back({std::vector<int>{1, 256, 1, 1}, v});
            k = 256;
        }
    
        std::vector<float> b(1);
        std::vector<float> w(1*k*1*1);
        ifs.read((char*)&b[0], sizeof(float)*b.size());
        ifs.read((char*)&w[0], sizeof(float)*w.size());
    
        go_params.push_back({std::vector<int>{1, k, 1, 1}, w});
        go_params.push_back({std::vector<int>{1, 1}, b});
    
        std::vector<float> b_w(board_count*board_moves);
        auto W_d = &b_w[0];
        for (int i=0; i<board_count; i++)
            for (int j=0; j<board_moves; j++)
                *(W_d++) = (i==j) ? 1 : 0;
        *(W_d-1) = 1;
    
        go_params.push_back({std::vector<int>{board_count, board_moves}, b_w});
    
        std::vector<float> fake_val_w(board_moves, 0);
        go_params.push_back({std::vector<int>{board_moves, 1}, fake_val_w});
    
        net.consume_params(go_params.begin());
        return true;
    }
    
    
    
    static void transpose_matrix(std::vector<float>& transpose, int rows, int cols)
    {
        auto orgin = transpose;
        for(int x = 0; x < rows; ++x) {
            for(int y = 0; y < cols; ++y){
                transpose[y*rows + x] = orgin[x*cols + y];
            }
        }
    }
    
    static void process_bn_beta(std::vector<float>& beta, const std::vector<float>& rv, const float epsilon = 1e-5) {
        for(auto i=0; i<beta.size(); i++) {
            beta[i] = beta[i] / std::sqrt(rv[i] + epsilon);
        }
    }
    
    
    bool load_leela_weights(leela_net_type& net, const std::string& path) {
        std::ifstream ifs(path);
        if (ifs.fail())
            return false;
        
        int channles = 0;
    
        int linecount = 0;
        std::string line;
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
        std::cerr << residual_blocks << " blocks" << std::endl;
    
        if (channles != 128 || residual_blocks != 6)
            throw std::runtime_error("not support leela weights.");
    
    
        // Re-read file and process
        ifs.clear();
        ifs.seekg(0, std::ios::beg);
    
        // Get the file format id out of the way
        std::getline(ifs, line);
    
    
        std::vector<param_data> params; 
    
        std::vector<float> stored_beta;
        std::vector<float> stored_mean;
    
        auto add_bn_running_var = [&params, &stored_beta, &stored_mean](std::vector<float>& rv) {
    
            int channels = (int)rv.size();
            auto shape = std::vector<int>{1, channels, 1, 1};
    
            // additional gamma, beta
            auto gamma = std::vector<float> (channels, 1);
            process_bn_beta(stored_beta, rv);
            params.push_back({shape, gamma});
            params.push_back({shape, stored_beta});
            params.push_back({shape, stored_mean});
            params.push_back({shape, rv});
        };
    
        auto add_fc_W = [&params](std::vector<float>& weights, int inputs, int outputs) {
    
            transpose_matrix(weights, outputs, inputs);
            params.push_back({std::vector<int>{inputs, outputs}, weights});
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
                    int insize = weights.size() / (channles*3*3);
                    params.push_back({std::vector<int>{channles, insize, 3, 3}, weights});
                } else if (linecount % 4 == 1) {
                    // bias
                    stored_beta = weights;
                } else if (linecount % 4 == 2) {
                    stored_mean = weights;
                } else if (linecount % 4 == 3) {
                    // var
                    add_bn_running_var(weights);
                } 
            } else if (linecount == plain_conv_wts) {
                // policy conv
                params.push_back({std::vector<int>{2, channles, 1, 1}, weights});
            } else if (linecount == plain_conv_wts + 1) {
                stored_beta = weights;
                
            } else if (linecount == plain_conv_wts + 2) {
                stored_mean = weights;
    
            } else if (linecount == plain_conv_wts + 3) {
                add_bn_running_var(weights);
    
            } else if (linecount == plain_conv_wts + 4) {
                add_fc_W(weights, 2*361, 362);
    
            } else if (linecount == plain_conv_wts + 5) {
                // fc b
                params.push_back({std::vector<int>{362}, weights});
                
            } else if (linecount == plain_conv_wts + 6) {
                //conv_val_w = std::move(weights);
                params.push_back({std::vector<int>{1, channles, 1, 1}, weights});
                
            } else if (linecount == plain_conv_wts + 7) {
                stored_beta = weights;
               
            } else if (linecount == plain_conv_wts + 8) {
                stored_mean = weights;
    
            } else if (linecount == plain_conv_wts + 9) {
                add_bn_running_var(weights);
    
            } else if (linecount == plain_conv_wts + 10) {
                add_fc_W(weights, 361, 256);
    
            } else if (linecount == plain_conv_wts + 11) {
                //std::copy(begin(weights), end(weights), begin(ip1_val_b));
                params.push_back({std::vector<int>{256}, weights});
    
            } else if (linecount == plain_conv_wts + 12) {
                add_fc_W(weights, 256, 1);
    
            } else if (linecount == plain_conv_wts + 13) {
                //std::copy(begin(weights), end(weights), begin(ip2_val_b));
                params.push_back({std::vector<int>{1}, weights});
            }
    
            linecount++;
        }
    
        net.consume_params(params.begin());
        return true;
    }
    
//////////////////////////////////////////////////////////

       

zero_model::zero_model()
: max_batch_size(32)
, zero_weights_loaded(false)
, leela_weights_loaded(false)
{}

void zero_model::set_batch_size(size_t size) {
    max_batch_size = size;
}

bool zero_model::load_weights(const std::string& path) {

    std::ifstream ifs(path, std::ifstream::binary);
    if (ifs.fail())
        return false;

    std::string key;
    bool is_leela_weights = false;
    bool is_zero_model = restore_string(ifs, key);

    if (!is_zero_model) {
        ifs.clear();
        ifs.seekg(0, std::ios::beg);
        int version;
        ifs >> version;
        if (version == 1)
            is_leela_weights = true;
    }
    ifs.close();

    if (is_zero_model) {
        zero_weights_loaded = load_zero_weights(zero_net, path);
        return zero_weights_loaded;
    } else if (is_leela_weights) {
        leela_weights_loaded = load_leela_weights(leela_net, path);
        return leela_weights_loaded;
    } else
        return load_dark_weights(dark_net, path);
}


const tensor& zero_model::forward(const tensor& input, double temperature, const tensor** value_out) {

    if (input.k() == 1) {
        return lightmodel::forward(dark_net, input, temperature, value_out);
    } else {
        return zero_weights_loaded ? 
                lightmodel::forward(zero_net, input, temperature, value_out) : 
                lightmodel::forward(leela_net, input, temperature, value_out);
    }
}


void zero_model::predict_batch_policies(const tensor& input, std::vector<prediction>::iterator it, double temperature) {

    int batch = input.num_samples();

    auto& out_tensor = forward(input, temperature, nullptr);
    auto src = out_tensor.host();

    for (int n=0;n<batch; n++) {
        it->resize(board_moves);
        std::copy(src, src+board_moves, it->begin());
        src += board_moves;
        it++;
    }
}

void zero_model::predict_batch(const tensor& input, std::vector<prediction_ex>::iterator it, double temperature) {

    int batch = input.num_samples();
    const tensor* v_tensor = nullptr;
    auto& out_tensor = forward(input, temperature, &v_tensor);
    auto src = out_tensor.host();
    auto data = v_tensor->host();

    for (int n=0;n<batch; n++) {
        it->first.resize(board_moves);
        std::copy(src, src+board_moves, it->first.begin());
        src += board_moves;
        it->second = data[n];
        it++;
    }
}

void zero_model::predict_batch_values(const tensor& input, std::vector<float>::iterator it) {

    int batch = input.num_samples();
    const tensor* v_tensor = nullptr;
    forward(input, 1, &v_tensor);
    auto data = v_tensor->host();

    for (int n=0;n<batch; n++) {
        (*it++) = data[n];
    }
}


}
