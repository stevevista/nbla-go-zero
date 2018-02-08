#pragma once

#include <nblapp/defs.hpp>
#include <memory>
#include <vector>

namespace nbla {

class CgVariable;
class NdArray;
typedef std::shared_ptr<CgVariable> CgVariablePtr;
typedef std::shared_ptr<NdArray> NdArrayPtr;

}

namespace nblapp {

using std::vector;
using nbla::CgVariablePtr;
using nbla::CgVariablePtr;
using nbla::NdArrayPtr;

class BaseSolver;

class NPP_API Variable {
public:
    Variable();
    Variable(const vector<int>& shape, const bool need_grad = false);
    Variable(const Variable&);
    Variable(Variable&&);

    Variable& operator =(const Variable&);
    Variable& operator =(Variable&&);

    Variable(CgVariablePtr);
    CgVariablePtr ptr() const;

    operator int() const = delete;
    operator float() const = delete;
    operator bool() const;
    Variable unlinked() const;

    int ndim() const;
    int dim(int i) const;
    int size() const;
    vector<int> shape() const;

    void set_need_grad(bool);
    bool need_grad() const;

    void set_persistent(bool);

    void reshape(const vector<int>& shape);

    static Variable constant(const vector<int>& shape, float fill);
    static Variable uniform(const vector<int>& shape, float low = 0, float high = 1);
    static Variable normal(const vector<int>& shape, float mu = 0, float sigma = 1);

    template<typename T>
    T* data();

    template<typename T>
    const T* data() const;

    template <typename T, typename ITER>
    void query(ITER begin, int copy_size=-1) const {
        if (copy_size < 0) copy_size = size();
        auto src = data<T>();
        std::copy(src, src+copy_size, begin);
    }

    template <typename T, typename ITER>
    void fill(ITER begin, ITER end) {
        auto dst = data<T>();
        std::copy(begin, end, dst);
    }


    void forward(bool clear_buffer = false,
                bool clear_no_need_grad = false);
    void backward(bool clear_buffer = false);
    void train_batch(BaseSolver& solver, float weight_decay = 0,
                bool clear_buffer = false,
                bool clear_no_need_grad = false);

    void fill(float v);

private:
    CgVariablePtr var_;
    NdArrayPtr bwd_grad_;
};

}

