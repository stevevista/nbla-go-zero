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

    void scale_columns (
        tensor& out,
        const tensor& m,
        const tensor& v
    );
    /*!
        requires
            - have_same_dimensions(out,m) == true
            - is_vector(v) == true
            - v.size() == mat(m).nc()
        ensures
            - performs: out = scale_columns(mat(m),mat(v));
    !*/

    void scale_rows (
        tensor& out,
        const tensor& m,
        const tensor& v
    );
    /*!
        requires
            - have_same_dimensions(out,m) == true
            - is_vector(v) == true
            - v.size() == m.num_samples()
        ensures
            - performs: out = scale_rows(mat(m),mat(v));
    !*/

    void scale_rows2 (
        float beta, 
        tensor& out,
        const tensor& m1,
        const tensor& m2,
        const tensor& v1,
        const tensor& v2
    );
    /*!
        requires
            - have_same_dimensions(out,m1) == true
            - have_same_dimensions(out,m2) == true
            - have_same_dimensions(v1,v2) == true
            - is_vector(v1) == true
            - v1.size() == m1.num_samples()
        ensures
            - performs: 
                out = beta*out + scale_rows(mat(m1) - scale_rows(mat(m2),mat(v1)), mat(v2));
    !*/

// ----------------------------------------------------------------------------------------
    
    void exp (
        tensor& dest,
        const tensor& src
    );
    /*!
        requires
            - dest.size() == src.size()
        ensures
            - performs: dest = exp(mat(src))
    !*/

// ----------------------------------------------------------------------------------------

    void log (
        tensor& dest,
        const tensor& src
    );
    /*!
        requires
            - dest.size() == src.size()
        ensures
            - performs: dest = log(mat(src))
    !*/

// ----------------------------------------------------------------------------------------

    void log10 (
        tensor& dest,
        const tensor& src
    );
    /*!
        requires
            - dest.size() == src.size()
        ensures
            - performs: dest = log10(mat(src))
    !*/

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
    /*!
        requires
            - dest does not alias the memory of lhs or rhs
            - The dimensions of lhs and rhs must be compatible for matrix multiplication.
              In particular:
                - Let L == trans_lhs ? trans(mat(lhs)) : mat(lhs)
                - Let R == trans_rhs ? trans(mat(rhs)) : mat(rhs)
                - Let D == mat(dest)
                - D.nr() == L.nr() && D.nc() == R.nc()
                  (i.e. dest must be preallocated and have the correct output dimensions)
                - L.nc() == R.nr()
        ensures
            - performs: dest = alpha*L*R + beta*mat(dest)
    !*/

// ----------------------------------------------------------------------------------------

    void multiply (
        bool add_to,
        tensor& dest,
        const tensor& src1,
        const tensor& src2
    );
    /*!
        requires
            - if (have_same_dimensions(dest, src1) == true) then
                - src2.num_samples() == 1
                - src2.nr() == 1
                - src2.nc() == 1
                - src2.k() == src1.k()
            - else
                - have_same_dimensions(src1, src2) == true) 
                - dest.num_samples() == 1
                - dest.nr() == 1
                - dest.nc() == 1
                - dest.k() == src1.k()
        ensures
            - Performs #dest == src1*src2 
              In particular, if the elements of dest, src1, and src2 were indexed by (n,k,r,c) then
              we would have:
                - if (have_same_dimensions(dest,src1)) then
                    #dest(n,k,r,c) == src1(n,k,r,c)*src2(k)
                - else
                    #dest(k) == sum over {n,r,c} of src1(n,k,r,c)*src2(n,k,r,c)
            - if (add_to) then
                - Instead of assigning the result to dest, this function adds the result to dest.
    !*/

    void multiply_zero_padded (
        bool add_to,
        tensor& dest,
        const tensor& src1,
        const tensor& src2
    );
    /*!
        ensures
            - if (add_to) then
                - performs: dest += src1 * src2
            - else
                - performs: dest = src1 * src2
            - In either case, the multiplication happens pointwise according to 4D tensor
              arithmetic.  If the dimensions don't match then missing elements are presumed
              to be equal to 0.
    !*/

// ----------------------------------------------------------------------------------------


    void affine_transform_range(
        size_t begin,
        size_t end,
        tensor& dest,
        const tensor& src1,
        const tensor& src2,
        const tensor& src3,
        const float A,
        const float B,
        const float C
    );

// ----------------------------------------------------------------------------------------

    void affine_transform_conv(
        tensor& dest,
        const tensor& src,
        const tensor& A,
        const tensor& B
    );
    /*!
        requires
            - have_same_dimensions(dest,src) == true
            - have_same_dimensions(A, B) == true
            - A.num_samples() == 1
            - A.nr() == 1
            - A.nc() == 1
            - A.k() == src.k()
        ensures
            - Performs #dest == A*src + B
              In particular, if the elements of dest and src were indexed by (n,k,r,c) then
              we would have:
                #dest(n,k,r,c) == A(k)*src(n,k,r,c) + B(k).
    !*/

// ----------------------------------------------------------------------------------------

    void compute_adam_update (
        size_t begin,
        size_t end,
        tensor& s,
        tensor& m,
        tensor& v,
        const float t,
        const float learning_rate,
        const float weight_decay,
        const float momentum1,
        const float momentum2,
        const tensor& params,
        const tensor& params_grad
    );
    /*!
        requires
            - s.size() == m.size() = v.size() == params.size() == params_grad.size()
            - t > 0
            - learning_rate > 0
            - weight_decay >= 0
            - 0 <= momentum1 < 1
            - 0 <= momentum2 < 1
            - begin <= end <= params.size()
        ensures
            - This function implements the ADAM parameter update method described in the paper:
                Kingma, Diederik P., and Jimmy Ba Adam. "A method for stochastic
                optimization." International Conference on Learning Representation. 2015.
              Specifically, it implements the method shown as Algorithm 1.
            - #s is the update vector that should be added to the parameters.
            - The function only operates in the half open range [begin,end) of the memory
              blocks of each tensor.  E.g. to make this function run on the entire tensor
              set begin to 0 and end to params.size().
    !*/

// -----------------------------------------------------------------------------------

    void threshold (
        tensor& data,
        float thresh
    );
    /*!
        ensures
            - Sets all elements of data to 1 or 0 depending on if they are above or below
              the given threshold.  Specifically, for all valid i:
                - #data.host()[i] == data.host()[i]>thresh ? 1 : 0
    !*/

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
        /*!
            requires
                - setup() has been called.  Specifically, setup() has been called like this:
                    this->setup(data, filters, stride_y, stride_x, padding_y, padding_x);
                - is_same_object(output,data) == false
                - is_same_object(output,filters) == false
                - filters.k() == data.k()
                - filters.nr() <= src.nr() + 2*padding_y
                - filters.nc() <= src.nc() + 2*padding_x
                - #output.num_samples() == data.num_samples()
                - #output.k() == filters.num_samples()
                - #output.nr() == 1+(data.nr() + 2*padding_y - filters.nr())/stride_y
                - #output.nc() == 1+(data.nc() + 2*padding_x - filters.nc())/stride_x
            ensures
                - Convolves filters over data.  If add_to_output==true then we add the
                  results to output, otherwise we assign to output, overwriting the
                  previous values in output.
                - filters contains filters.num_samples() filters. 
        !*/

        void operator() (
            const bool add_to_output,
            resizable_tensor& output,
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

    class pooling
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The pooling object is a tool for performing spatial pooling over a tensor.
                It can be configured to do either max or average pooling.
        !*/
    public:

        pooling(const pooling&) = delete;
        pooling& operator=(const pooling&) = delete;

        pooling (
        ) = default;

        void clear(
        ) { impl.clear(); }

        void setup_max_pooling(
            int window_height,
            int window_width,
            int stride_y,
            int stride_x,
            int padding_y,
            int padding_x
        ) { impl.setup_max_pooling(window_height, window_width, stride_y, stride_x, padding_y, padding_x); }
        /*!
            requires
                - window_height > 0
                - window_width > 0
                - stride_y > 0
                - stride_x > 0
                - 0 <= padding_y < window_height
                - 0 <= padding_x < window_width
            ensures
                - When you call operator() it will do max pooling with the given
                  parameters.
        !*/

        void setup_avg_pooling(
            int window_height,
            int window_width,
            int stride_y,
            int stride_x,
            int padding_y,
            int padding_x
        ) { impl.setup_avg_pooling(window_height, window_width, stride_y, stride_x, padding_y, padding_x); }
        /*!
            requires
                - window_height > 0
                - window_width > 0
                - stride_y > 0
                - stride_x > 0
                - 0 <= padding_y < window_height
                - 0 <= padding_x < window_width
            ensures
                - When you call operator() it will do average pooling with the given
                  parameters.
        !*/

        bool does_max_pooling(
        ) const { return impl.does_max_pooling(); }

        void operator() (
            resizable_tensor& dest,
            const tensor& src
        ) { impl(dest, src); }
        /*!
            requires
                - is_same_object(dest,src) == false
                - either setup_max_pooling() or setup_avg_pooling() has been called.
                - window_width  <= src.nc() + 2*padding_x
                - window_height <= src.nr() + 2*padding_y
            ensures
                - #dest.num_samples() == src.num_samples()
                - #dest.k() == src.k()
                - #dest.nr() == 1 + (src.nr() + 2*padding_y - window_height)/stride_y
                - #dest.nc() == 1 + (src.nc() + 2*padding_x - window_width)/stride_x
                - WINDOW == centered_rect(x*stride_x + window_width/2 - padding_x,
                                          y*stride_y + window_height/2 - padding_y,
                                          window_width,
                                          window_height)
                - for all valid s, k, r, and c:
                    - if (does_max_pooling()) then
                        - image_plane(#dest,s,k)(r,c) == max(subm_clipped(image_plane(src,s,k),WINDOW(c,r)))
                    - else
                        - image_plane(#dest,s,k)(r,c) == mean(subm_clipped(image_plane(src,s,k),WINDOW(c,r)))
        !*/

        private:
#ifdef DLIB_USE_CUDA
        cuda::pooling impl;
#else
        cpu::pooling impl;
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

    void prelu (
        tensor& dest,
        const tensor& src,
        const tensor& param
    );
    /*!
        requires
            - have_same_dimensions(dest, src) == true
            - param.size() == 1
        ensures
            - for all valid i:
                - if (src.host()[i] > 0) then
                    - #dest.host()[i] == src.host()[i]
                - else
                    - #dest.host()[i] == src.host()[i] * param.host()[0]
            - This function supports in-place operation, i.e. having
              is_same_object(dest, src)==true
    !*/

// ----------------------------------------------------------------------------------------

    void tanh (
        tensor& dest,
        const tensor& src
    );


// ----------------------------------------------------------------------------------------

    void copy_tensor(
            bool add_to,
            tensor& dest,
            size_t dest_k_offset,
            const tensor& src,
            size_t src_k_offset,
            size_t count_k
    );
    /*!
        requires
            - dest.nc() == src.nc()
            - dest.nr() == src.nr()
            - dest.num_samples() == src.num_samples()
            - dest.k() - dest_k_offset >= count_k
            - src.k() - src_k_offset >= count_k
            - is_same_object(dest,src) == false
            - The memory areas of src and dest do not overlap.
        ensures
            - if (add_to) then
                - performs: dest[i, k + dest_k_offset, r, c] += src[i, k + src_k_offset, r, c], where k in [0..count_k]
                  i.e., adds content of each sample from src in to corresponding place of sample at dest.
            - else
                - performs: dest[i, k + dest_k_offset, r, c]  = src[i, k + src_k_offset, r, c], where k in [0..count_k]
                  i.e., copies content of each sample from src in to corresponding place of sample at dest.
    !*/

// ----------------------------------------------------------------------------------------

}}

#ifdef NO_MAKEFILE
#include "tensor_tools.cpp"
#endif

#endif // DLIB_TeNSOR_TOOLS_H_


