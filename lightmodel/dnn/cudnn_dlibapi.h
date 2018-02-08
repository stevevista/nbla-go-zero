// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuDNN_H_
#define DLIB_DNN_CuDNN_H_

#ifdef DLIB_USE_CUDA

#include "cuda_errors.h"
#include <memory>
#include "cuda_data_ptr.h"

namespace dlib
{
    class tensor;

    namespace cuda 
    {

    // -----------------------------------------------------------------------------------

        class tensor_descriptor
        {
            /*!
                Each tensor object will carry a tensor_descriptor in it when compiled with
                CUDA.
            !*/

        public:
            // not copyable
            tensor_descriptor(const tensor_descriptor&) = delete;
            tensor_descriptor& operator=(const tensor_descriptor&) = delete;
            // but is movable
            tensor_descriptor(tensor_descriptor&& item) : tensor_descriptor() { swap(item); }
            tensor_descriptor& operator=(tensor_descriptor&& item) { swap(item); return *this; }

            tensor_descriptor();
            ~tensor_descriptor();

            void set_size(
                int n, 
                int k,
                int nr, 
                int nc 
            );
            /*!
                ensures
                    - if any of the arguments are 0 then they are all set to 0 in the tensor.
            !*/

            void get_size (
                int& n, 
                int& k,
                int& nr,
                int& nc 
            ) const;

            const void* get_handle (
            ) const { return handle; }

        private:

            void swap(tensor_descriptor& item) { std::swap(handle, item.handle); }

            void* handle;
        };

        // ------------------------------------------------------------------------------------

        void add(
            float beta,
            tensor& dest,
            float alpha,
            const tensor& src
        );

    // ------------------------------------------------------------------------------------

        class tensor_conv
        {
        public:
            tensor_conv(const tensor_conv&) = delete;
            tensor_conv& operator=(const tensor_conv&) = delete;

            tensor_conv();

            void clear(
            );

            ~tensor_conv (
            );

            void operator() (
                const bool add_to_output,
                tensor& output,
                const tensor& data,
                const tensor& filters
            );

           void setup(
                const tensor& data,
                const tensor& filters,
                int stride_y,
                int stride_x,
                int padding_y,
                int padding_x
            );

        private:

            // These variables record the type of data given to the last call to setup().
            int stride_y;
            int stride_x;
            int padding_y;
            int padding_x;
            long data_num_samples, data_k, data_nr, data_nc;
            long filters_num_samples, filters_k, filters_nr, filters_nc;


            void* filter_handle;
            void* conv_handle;

            // dimensions of the output tensor from operator()
            int out_num_samples;
            int out_k;
            int out_nr;
            int out_nc;

            int forward_algo;

            size_t forward_workspace_size_in_bytes;
            std::shared_ptr<resizable_cuda_buffer> workspace;
            cuda_data_void_ptr forward_workspace;
        };


    // ------------------------------------------------------------------------------------

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
                  each location.  That is, softmax() outputs a new tensor, #dest, where
                  each of the spatial locations in dest (i.e. image idx, row idx, and
                  column idx) contains the output of s() evaluated over the channel values
                  at each location.
                - This function supports in-place operation, i.e. having
                  is_same_object(dest, src)==true
        !*/

    // ------------------------------------------------------------------------------------

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

    // ------------------------------------------------------------------------------------

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

    // ------------------------------------------------------------------------------------

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

    // ------------------------------------------------------------------------------------

    } 
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuDNN_H_

