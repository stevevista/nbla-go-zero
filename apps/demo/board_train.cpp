

#include <nblapp/parameter.hpp>
#include <nblapp/ops.hpp>
#include <nblapp/solver.hpp>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_INLINE // We need this define to avoid multiple definition
#include "stb/stb_image.h"



void load_image_data(const std::string& filename, float*& dst) {
    int w, h, c;
    unsigned char *data = stbi_load(filename.c_str(), &w, &h, &c, 3);
    for (int k=0; k<3; k++) {
        for (int y=0; y<h; y++) {
            for (int x=0; x<w; x++) {
                unsigned char v = data[(y*w+x)*3 +k];
                (*dst++) = (float)v/255;
            }
        }
    }
    free(data);
}

void load_lable(std::istream& is, float*& dst) {
    for (int i=0; i<364; i++) {
        is >> (*dst++);
    }
}


using namespace nblapp;

Variable bn_conv2d(const string& name, const Variable& x, 
    const int filters, const std::vector<int>& kernel, const std::vector<int>& stride,
    bool batch_stat=true) {
        
    auto h = nn::conv2d(name, x, filters, kernel, stride, PAD_VALID, false);
    return nn::batchnorm(name, h, batch_stat);
}


Variable train_model( const Variable& x, const Variable& l) {
    auto h = nn::maxpool(nn::relu(bn_conv2d("L0", x, 64, {7, 7}, {2,2}, true)), {2,2}, {2,2});
    h = nn::maxpool(nn::relu(bn_conv2d("L1", h, 128, {3, 3}, {1,1}, true)), {2,2}, {2,2});
    h = nn::maxpool(nn::relu(bn_conv2d("L2", h, 256, {3, 3}, {1,1}, true)), {2,2}, {2,2});
    h = nn::relu(bn_conv2d("L3", h, 512, {3, 3}, {1,1}, true));
    h = nn::fully_connected("L4", h, 364);

    h = nn::sigmoid_cross_entropy(h, l);
    return nn::reduce_mean(h);
}

int main() {
    float weight_decay = 0.0001;
    float learning_rate = 0.1;

    char buf[1024];

    auto x = Variable({32,3,400,400});
    auto l = Variable({32,364});
    auto loss = train_model(x, l);
    
    auto solver = MomentumSolver(learning_rate, 0.9);

    std::ifstream ifs("/home/steve/dev/data1/lable.txt");

    int data_index = 0;
    while(data_index < 10000 - 32)
    {

        auto x_D = x.data<float>();
        auto l_D = l.data<float>();
        for (int b=0; b<32; b++) {
            sprintf(buf, "/home/steve/dev/data1/b_%06d.jpg", data_index++);
            load_image_data(buf, x_D);
            load_lable(ifs, l_D);
        }

        loss.train_batch(solver, weight_decay);
        
        const auto this_loss = loss.data<float>()[0];
    
    
        std::cout << data_index << ": " << this_loss << std::endl;

    }


    

    

    
}
