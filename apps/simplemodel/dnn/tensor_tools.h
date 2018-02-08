// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TeNSOR_TOOLS_H_
#define DLIB_TeNSOR_TOOLS_H_

#include "tensor.h"
#include "cudnn_dlibapi.h"
#include "cublas_dlibapi.h"
#include "cpu_dlib.h"
#include "cuda_dlib.h"
#include <memory>

namespace dlib
{
    bool dnn_prefer_fastest_algorithms();
    void set_dnn_prefer_fastest_algorithms();
    void set_dnn_prefer_smallest_algorithms();
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
    );
// ----------------------------------------------------------------------------------------

    void affine_transform_conv(
        tensor& dest,
        const tensor& src,
        const tensor& A,
        const tensor& B
    );

// ----------------------------------------------------------------------------------------

    void add(
        float beta,
        tensor& dest,
        float alpha,
        const tensor& src
    );
    /*!
        requires
            - One of the following is true: 
                - have_same_dimensions(src, dest)
                - src.num_samples()==1 && src.k()==dest.k() && src.nr()==1 && src.nc()==1
                - src.num_samples()==1 && src.k()==dest.k() && src.nr()==dest.nr() && src.nc()==dest.nc()
                - src.num_samples()==1 && src.k()==1 && src.nr()==dest.nr() && src.nc()==dest.nc()
                - src.num_samples()==dest.num_samples() && src.k()==1 && src.nr()==1 && src.nc()==1
            - is_same_object(src,dest) == false
        ensures
            - performs: dest = beta*dest + alpha*src
              However, how the addition happens depends on the dimensions of src.  In
              particular, this function adds the scaled values of one src tensor to dest.
              Each dimension of the src tensor must match the corresponding dimension of
              the dest tensor or must be equal to 1. In the latter case, the same value
              from the src tensor, for those dimensions, will be used to add into the dest
              tensor.
    !*/

// ----------------------------------------------------------------------------------------

    void add (
        tensor& dest,
        const tensor& src1,
        const tensor& src2
    );
    /*!
        ensures
            - performs: dest = src1 + src2
              The addition happens pointwise according to 4D tensor arithmetic.  If the
              dimensions don't match then missing elements are presumed to be equal to 0.
    !*/

// ----------------------------------------------------------------------------------------

    class tensor_conv
    {
    public:
        tensor_conv(const tensor_conv&) = delete;
        tensor_conv& operator=(const tensor_conv&) = delete;

        tensor_conv() {}

        void clear(
        ) { impl.clear(); }

        void operator() (
            const bool add_to_output,
            tensor& output,
            const tensor& data,
            const tensor& filters
        ) { impl(add_to_output,output,data,filters); }
 
        void setup(
            const tensor& data,
            const tensor& filters,
            int stride_y,
            int stride_x,
            int padding_y,
            int padding_x
        ) {impl.setup(data,filters,stride_y,stride_x,padding_y,padding_x); }
        /*!
            requires
                - filters.k() == data.k()
                - stride_y > 0
                - stride_x > 0
                - 0 <= padding_y < filters.nr()
                - 0 <= padding_x < filters.nc()
            ensures
                - When operator() is called, the output tensor will have these dimensions:
                    - output.nr() == 1+(data.nr() + 2*padding_y - filters.nr())/stride_y
                    - output.nc() == 1+(data.nc() + 2*padding_x - filters.nc())/stride_x
                    - output.num_samples() == data.num_samples()
                    - output.k() == filters.num_samples()
                - The point of setup() is to allow this object to gather information about
                  all the tensor sizes and filter layouts involved in the computation.  In
                  particular, the reason the tensors are input into setup() is just to
                  observe their sizes.  setup() doesn't do anything with the contents of
                  the tensors, or store any kind of references to the data or filter
                  tensors. 
        !*/
       
    private:
#ifdef DLIB_USE_CUDA
        cuda::tensor_conv impl;
#else
        cpu::tensor_conv impl;
#endif

    };

// ----------------------------------------------------------------------------------------

    void softmax (
        tensor& dest,
        const tensor& src
    );
    /*!
        requires
            - have_same_dimensions(dest, src) == true
        ensures
            - Note that the softmax function is a vector valued function: 
                s(x) == exp(x)/sum(exp(x)) 
            - Computes the softmax function on src and writes the results to dest.  The
              softmax is computed per spatial location across the different channels at
              each location.  That is, softmax() outputs a new tensor, #dest, where each of
              the spatial locations in dest (i.e. image idx, row idx, and column idx)
              contains the output of s() evaluated over the channel values at each
              location.
            - This function supports in-place operation, i.e. having
              is_same_object(dest, src)==true
    !*/

// ----------------------------------------------------------------------------------------

    void sigmoid (
        tensor& dest,
        const tensor& src
    );
    /*!
        requires
            - have_same_dimensions(dest, src) == true
        ensures
            - for all valid i:
                - #dest.host()[i] == 1/(1+std::exp(-src.host()[i])) 
            - This function supports in-place operation, i.e. having
              is_same_object(dest, src)==true
    !*/

// ----------------------------------------------------------------------------------------

    void relu (
        tensor& dest,
        const tensor& src
    );
    /*!
        requires
            - have_same_dimensions(dest, src) == true
        ensures
            - for all valid i:
                - #dest.host()[i] == std::max(0,src.host()[i]) 
            - This function supports in-place operation, i.e. having
              is_same_object(dest, src)==true
    !*/

// ----------------------------------------------------------------------------------------

    void tanh (
        tensor& dest,
        const tensor& src
    );
    /*!
        requires
            - have_same_dimensions(dest, src) == true
        ensures
            - for all valid i:
                - #dest.host()[i] == std::tanh(src.host()[i]) 
            - This function supports in-place operation, i.e. having
              is_same_object(dest, src)==true
    !*/

// ----------------------------------------------------------------------------------------

}}

#ifdef NO_MAKEFILE
#include "tensor_tools.cpp"
#endif

#endif // DLIB_TeNSOR_TOOLS_H_


