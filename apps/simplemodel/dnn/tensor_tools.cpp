// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TeNSOR_TOOLS_CPP_
#define DLIB_TeNSOR_TOOLS_CPP_

#include "tensor_tools.h"
#include <atomic>

namespace dlib
{
    namespace
    {
        std::atomic<bool>& dnn_prefer_fastest_algo (
        )
        {
            static std::atomic<bool> var(true);
            return var;
        }
    }

    bool dnn_prefer_fastest_algorithms (
    )
    {
        return dnn_prefer_fastest_algo();
    }

    void set_dnn_prefer_fastest_algorithms(
    )
    {
        dnn_prefer_fastest_algo() = true;
    }

    void set_dnn_prefer_smallest_algorithms(
    )
    {
        dnn_prefer_fastest_algo() = false;
    }
}

namespace dlib { namespace tt
{

// ----------------------------------------------------------------------------------------

    void gemm (
        float beta,
        tensor& dest,
        float alpha,
        const tensor& lhs,
        bool trans_lhs,
        const tensor& rhs,
        bool trans_rhs
    )
    {
#ifdef DLIB_USE_CUDA
        cuda::gemm(beta, dest, alpha, lhs, trans_lhs, rhs, trans_rhs);
#else
        if (beta != 0)
        {
            if (trans_lhs && trans_rhs)
                dest = alpha*trans(mat(lhs))*trans(mat(rhs)) + beta*mat(dest);
            else if (!trans_lhs && trans_rhs)
                dest = alpha*mat(lhs)*trans(mat(rhs)) + beta*mat(dest);
            else if (trans_lhs && !trans_rhs)
                dest = alpha*trans(mat(lhs))*mat(rhs) + beta*mat(dest);
            else
                dest = alpha*mat(lhs)*mat(rhs) + beta*mat(dest);
        }
        else
        {
            if (trans_lhs && trans_rhs)
                dest = alpha*trans(mat(lhs))*trans(mat(rhs));
            else if (!trans_lhs && trans_rhs)
                dest = alpha*mat(lhs)*trans(mat(rhs));
            else if (trans_lhs && !trans_rhs)
                dest = alpha*trans(mat(lhs))*mat(rhs);
            else
                dest = alpha*mat(lhs)*mat(rhs);
        }
#endif
    }

// ----------------------------------------------------------------------------------------

    void affine_transform_conv(
        tensor& dest,
        const tensor& src,
        const tensor& A,
        const tensor& B
    )
    {
#ifdef DLIB_USE_CUDA
        cuda::affine_transform_conv(dest,src,A,B);
#else
        cpu::affine_transform_conv(dest,src,A,B);
#endif
    }

// ----------------------------------------------------------------------------------------

    void add(
        float beta,
        tensor& dest,
        float alpha,
        const tensor& src
    )
    {
#ifdef DLIB_USE_CUDA
        cuda::add(beta,dest,alpha,src);
#else
        cpu::add(beta,dest,alpha,src);
#endif
    }

// ----------------------------------------------------------------------------------------

    void add (
        tensor& dest,
        const tensor& src1,
        const tensor& src2
    )
    {
#ifdef DLIB_USE_CUDA
        cuda::add(dest, src1, src2);
#else
        cpu::add(dest, src1, src2);
#endif
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    void softmax (
        tensor& dest,
        const tensor& src
    )
    {
#ifdef DLIB_USE_CUDA
        cuda::softmax(dest,src);
#else
        cpu::softmax(dest,src);
#endif
    }

// ----------------------------------------------------------------------------------------

    void sigmoid (
        tensor& dest,
        const tensor& src
    )
    {
#ifdef DLIB_USE_CUDA
        cuda::sigmoid(dest,src);
#else
        cpu::sigmoid(dest,src);
#endif
    }

// ----------------------------------------------------------------------------------------

    void relu (
        tensor& dest,
        const tensor& src
    )
    {
#ifdef DLIB_USE_CUDA
        cuda::relu(dest,src);
#else
        cpu::relu(dest,src);
#endif
    }

// ----------------------------------------------------------------------------------------

    void tanh (
        tensor& dest,
        const tensor& src
    )
    {
#ifdef DLIB_USE_CUDA
        cuda::tanh(dest,src);
#else
        cpu::tanh(dest,src);
#endif
    }

// ----------------------------------------------------------------------------------------

}}

#endif // DLIB_TeNSOR_TOOLS_CPP_

