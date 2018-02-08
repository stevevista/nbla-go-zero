#pragma once

#include <nblapp/defs.hpp>
#include <cstdint>

namespace nblapp {


enum Engine {
    CPU = 1,
    CUDA = 2,
    CUDNN = 4
};

namespace context {

class NPP_API Scope {
    uint32_t prev_context;
public:
    Scope(const uint32_t eng);
    ~Scope();
};

NPP_API void set_default_context(const uint32_t eng);

}
}
