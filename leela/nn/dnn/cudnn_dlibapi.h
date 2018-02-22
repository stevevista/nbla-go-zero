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
    class resizable_tensor;

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
        /*!
            requires
                - One of the following is true: 
                    - have_same_dimensions(src, dest)
                    - src.num_samples()==1 && src.k()==dest.k() && src.nr()==1 && src.nc()==1
                    - src.num_samples()==1 && src.k()==dest.k() && src.nr()==dest.nr() && src.nc()==dest.nc()
                    - src.num_samples()==1 && src.k()==1 && src.nr()==dest.nr() && src.nc()==dest.nc()
                - is_same_object(src,dest) == false
            ensures
                - performs: dest = beta*dest + alpha*src
                  However, how the addition happens depends on the dimensions of src.  In
                  particular, this function adds the scaled values of one src tensor to
                  dest. Each dimension of the src tensor must match the corresponding
                  dimension of the dest tensor or must be equal to 1. In the latter case,
                  the same value from the src tensor, for those dimensions, will be used to
                  add into the dest tensor.
        !*/

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

            void operator() (
                const bool add_to_output,
                resizable_tensor& output,
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

        class pooling
        {
        public:

            pooling(const pooling&) = delete;
            pooling& operator=(const pooling&) = delete;

            pooling (
            );

            ~pooling(
            );

            void clear(
            );

            void setup_max_pooling(
                int window_height,
                int window_width,
                int stride_y,
                int stride_x,
                int padding_y,
                int padding_x
            );

            void setup_avg_pooling(
                int window_height,
                int window_width,
                int stride_y,
                int stride_x,
                int padding_y,
                int padding_x
            );

            bool does_max_pooling(
            ) const { return do_max_pooling; }

            void operator() (
                resizable_tensor& dest,
                const tensor& src
            );

        private:

            void setup(
                int window_height,
                int window_width,
                int stride_y,
                int stride_x,
                int padding_y,
                int padding_x,
                int pooling_mode
            );

            void* handle;
            int window_height;
            int window_width;
            int stride_y;
            int stride_x;
            int padding_y;
            int padding_x;
            bool do_max_pooling;
        };
    // ------------------------------------------------------------------------------------

        void softmax (
            tensor& dest,
            const tensor& src
        );

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

