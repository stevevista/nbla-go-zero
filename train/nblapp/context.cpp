
#include "context.hpp"
#include <nbla/exception.hpp>
#include <nbla/init.hpp>
#include <nbla/cuda/init.hpp>
#include <nbla/cuda/cudnn/init.hpp>
#include <nbla/context.hpp>
#include <iostream>


namespace nblapp {

using namespace nbla;


class Initializer {
public:
    Initializer() {
        std::cerr << "Initializing CPU extension..." << std::endl;
        init_cpu();
        std::cerr << "Initializing CUDA/CUDNN extension..." << std::endl;

        #ifdef WITH_CUDNN
        init_cuda();
        init_cudnn();
        context::set_default_context(CPU|CUDA|CUDNN);
        #endif
    }
};


Context current_ctx;
static uint32_t current_ctx_eng;

namespace context {


static int context_level = 0;


static void set_context(const uint32_t eng) {

    current_ctx_eng = eng;

    switch (current_ctx_eng) {
    
    case (CUDA):
        current_ctx = Context{"cuda", "CudaArray", "0", "default"};
        break;
    case (CUDNN):
        current_ctx = Context{"cuda", "CudaArray", "0", "cudnn"};
        break;
    case (CPU|CUDA):
        current_ctx = Context{"cpu|cuda", "CudaArray", "0", "default"};
        break;
    case (CPU|CUDNN):
        current_ctx = Context{"cpu|cuda", "CudaArray", "0", "cudnn"};
        break;
    case (CPU|CUDA|CUDNN):
        current_ctx = Context{"cpu|cuda", "CudaArray", "0", "default|cudnn"};
        break;
    case (CPU):
    default:
        current_ctx = Context{};
        break;
    }
}

Scope::Scope(const uint32_t eng) {
    context_level++;
    prev_context = current_ctx_eng;
    set_context(eng);
}

Scope::~Scope() {
    context_level--;
    set_context(prev_context);
}


void set_default_context(const uint32_t eng) {
    NBLA_CHECK(context_level == 0, error_code::unclassified, "It cannot be called inside any context_scope.");
    set_context(eng);
}


static Initializer global_initializer;

}
}

