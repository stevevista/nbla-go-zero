#include <iostream>
#include "../model.h"
#include "graph.pb.h"
#include <fstream>

using namespace std;
using namespace google;

std::string space(int indent) {
    std::vector<char> s(indent*2+1, ' ');
    s.back() = 0;
    return &s[0];
}


std::string to_str(const tensorflow::TensorShapeProto& s) {
    if (s.unknown_rank())
        return "unknown_rank";

    std::ostringstream sout; 
    sout << "[";
    for (auto& d : s.dim()) {
        sout << d.size() << ", ";
    }
    sout << "]";
    return sout.str();
}


std::string to_str(const tensorflow::TensorProto& t) {

    std::ostringstream sout; 
    int cnt_size = t.tensor_content().size();
    sout << DataType_Name(t.dtype());
    sout << " ";
    sout << to_str(t.tensor_shape());
    sout << " " << cnt_size;
    sout << " " << t.float_val_size();
    if (cnt_size > 0 &&  cnt_size <= 32) {
        if (t.dtype() == tensorflow::DT_INT32) {
            std::vector<int> tmp(cnt_size/4);
            memcpy(&tmp[0], &t.tensor_content()[0], cnt_size);
            sout << " -- ";
            for (auto i : tmp) sout << i << ",";
        }
    }

    return sout.str();
}

std::string to_str(const tensorflow::AttrValue::ListValue& list) {
    std::ostringstream sout; 
    if (list.s_size()) {
        for (auto& s : list.s()) sout << s << ", ";
    }
    if (list.i_size()) {
        for (auto& s : list.i()) sout << s << ", ";
    }

    return sout.str();
}


void print(const tensorflow::AttrValue& val, int indent) {
    auto vcase = val.value_case();
    switch (vcase) {
    case 2:
        std::cout << space(indent) << "string (" << val.s() << ")" << std::endl;
        break;
    case 3:
        std::cout << space(indent) << "int (" << val.i() << ")" << std::endl;
        break;
    case 4:
        std::cout << space(indent) << "float (" << val.f() << ")" << std::endl;
        break;
    case 5:
        std::cout << space(indent) << "bool (" << val.b() << ")" << std::endl;
        break;
    case 6:
        std::cout << space(indent) << "type (" << DataType_Name(val.type()) << ")" << std::endl;
        break;
    case 7:
        std::cout << space(indent) << "shape (" << to_str(val.shape()) << ")" << std::endl;
        break;
    case 8:
        std::cout << space(indent) << "tensor (" << to_str(val.tensor()) << ")" << std::endl;
        break;
    case 1:
        std::cout << space(indent) << "list (" << to_str(val.list()) << ")" << std::endl;
        break;
    case 10:
        std::cout << space(indent) << "function" << std::endl;
        break;
    case 9:
        std::cout << space(indent) << "placeholder (" << val.placeholder() << ")" << std::endl;
        break;
    default:
        std::cout << space(indent) << "unknown type" << std::endl;
    }
}



void print(const tensorflow::NodeDef& node, int indent) {
    std::cout << space(indent) << "[" << node.name() << "]" << std::endl;
    std::cout << space(indent+1) << "op: " << node.op() << std::endl;
    if (node.input_size()) {
        std::cout << space(indent+1) << "input: ";
        for (auto& c : node.input())
            std::cout << c << ", "; 
        std::cout << std::endl; 
    }
    for (auto& pair : node.attr()) {
        std::cout << space(indent+1) << pair.first << ":" << std::endl; 
        print(pair.second, indent+2);
    }

    std::cout << std::endl; 
}


void print(const tensorflow::GraphDef& g, int indent) {

    for (auto& node: g.node())
        print(node, indent+1);
}

static bool end_with(const std::string& s, const std::string& sub) {
    auto pos = s.rfind(sub);
    return pos != std::string::npos && pos == s.size() - sub.size();
}

namespace lightmodel {

int get_shape(const tensorflow::TensorShapeProto& s, std::vector<int>& shape) {
        
    shape.clear();
    if (s.unknown_rank())
        return 0;
 
    int size = 1;
    for (auto& d : s.dim()) {
        shape.push_back(d.size());
        size *= d.size();
    }

    return size;
}

bool get_tensor(const tensorflow::NodeDef& node, std::vector<int>& shape, std::vector<float>& w) {

    for (auto& pair : node.attr()) {
        if (pair.second.value_case() == 8) {
            auto& t = pair.second.tensor();
            if (t.dtype() != tensorflow::DT_FLOAT)
                throw std::runtime_error("not DT_FLOAT tensor");

            int size = get_shape(t.tensor_shape(), shape);
            if (size == 0)
                throw std::runtime_error("tensor size cannot be 0");

            bool need_transpose = false;
            int out, in, nr, nc;
            if (shape.size() == 4) {
                // tf: nr, nc, in, out
                // to: out, in, nr, nc
                need_transpose = true;
                nr = shape[0];
                nc = shape[1];
                in = shape[2];
                out = shape[3];
                shape = std::vector<int>{out, in, nr, nc};
            }

            
            if (t.tensor_content().size() == size*sizeof(float)) {
                float* src = (float*)&t.tensor_content()[0];
                if (need_transpose) {
                    for (int i=0; i < out; i++) {
                        for (int j=0; j < in; j++) {
                            for (int m=0; m < nr; m++) {
                                for (int n=0; n < nc; n++) {
                                    w.push_back(src[m*(nc*in*out)+n*(in*out)+j*out+i]);
                                }
                            }
                        }
                    }
                } else {
                    w.resize(size);
                    std::copy(src, src+size, w.begin());
                }
            }
            else if (t.float_val_size() == size) {
                if (need_transpose) {
                    for (int i=0; i < out; i++) {
                        for (int j=0; j < in; j++) {
                            for (int m=0; m < nr; m++) {
                                for (int n=0; n < nc; n++) {
                                    w.push_back(t.float_val(m*(nc*in*out)+n*(in*out)+j*out+i));
                                }
                            }
                        }
                    }
                 } else {
                    w.resize(size);
                    std::copy(t.float_val().begin(), t.float_val().end(), w.begin());
                }
                
            } else {
                throw std::runtime_error("tensor data size 0");
            }
            
            return true;
        }
    }

    return false;
}

bool parse_graph(tensorflow::GraphDef& g, const std::string& path) {
    
    ifstream ifs(path, ifstream::binary);
    if (!ifs)
        return false;

    ifs.seekg(0, std::ios::end);    // go to the end  
    auto length = ifs.tellg();           // report location (this is the length)  
    ifs.seekg(0, std::ios::beg);    // go back to the beginning  
    std::vector<uint8_t> buffer(length);
    ifs.read((char*)&buffer[0], length);       // read the whole file into the buffer  
    ifs.close();                    // close file handle 

    protobuf::io::CodedInputStream coded_stream(&buffer[0], length);
    coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);

    if (!g.ParseFromCodedStream(&coded_stream)) {
        return false;
    }
    return true;
}

bool load_ago_policy(alphago::net_type& net, const std::string& path) {

    tensorflow::GraphDef g;

    if (!parse_graph(g, path)) {
        return false;
    }
    //print(g, 0);

    std::vector<param_data> params(22*2+1+2); 

    for (auto& node: g.node()) {

        auto& name = node.name();
        std::vector<int> shape;
        std::vector<float> w;
        int level;
        int offset;
        int delta = 0;

        if (name.find("conv") == 0) {
            level = std::stoi(name.substr(4));
        }
        else if (name == "fc/weight") {
            level = 22;
        }
        else if (name.find("fc/") == 0) {
            delta = -1; // before is a conv_no_bias
            level = 23;
        }

        if (end_with(name, "/bias")) {

            if (!get_tensor(node, shape, w))
                throw std::runtime_error(name + ": tensor not found");

            offset = 1;

            if (name.find("fc/") == 0) {

                std::vector<float> b_w(361*361);
                auto W_d = &b_w[0];
                for (int i=0; i<361; i++)
                    for (int j=0; j<361; j++)
                        *(W_d++) = (i==j) ? 1 : 0;
                //*(W_d-1) = 0;

                //w.resize(362);
               // w.back() = 0;
                //shape[0] = 362;
                params[22*2+1+0] = {std::vector<int>{361, 361}, b_w};
            }
            
        }
        else if (end_with(name, "/weight")) {

            if (!get_tensor(node, shape, w))
                throw std::runtime_error(name + ": tensor not found");

            offset = 0;
        }

        if (w.size())
            params[level*2+offset+delta] = {shape, w}; 
    }

    net.consume_params(params.begin());
    return true;
}


bool load_ago_value(alphago::vnet_type& net, const std::string& path) {
    
    tensorflow::GraphDef g;
    
    if (!parse_graph(g, path)) {
        return false;
    }
    
    std::vector<param_data> params(20*2); 
    
    for (auto& node: g.node()) {
        
        auto& name = node.name();
        std::vector<int> shape;
        std::vector<float> w;
        int level;
        int offset;

        if (name.find("vconv") == 0) {
            level = std::stoi(name.substr(5));
        }
        else if (name.find("fc") == 0) {
            level = std::stoi(name.substr(2)) + 17;
        }

        if (end_with(name, "/bias")) {
    
            if (!get_tensor(node, shape, w))
                throw std::runtime_error(name + ": tensor not found");
    
            offset = 1;
        }
        else if (end_with(name, "/weight")) {
 
            if (!get_tensor(node, shape, w))
                throw std::runtime_error(name + ": tensor not found");
    
            offset = 0;
        }

        if (w.size())
            params[level*2+offset] = {shape, w}; 
    }
    
    net.consume_params(params.begin());
    return true;
}

}

