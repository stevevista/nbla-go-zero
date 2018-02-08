// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CPU_H_
#define DLIB_DNN_CPU_H_

// This file contains CPU implementations of the GPU based functions in cuda_dlib.h
// and cudnn_dlibapi.h

#include "tensor.h"

namespace dlib
{
    namespace cpu 
    {
        void add(
            float beta,
            tensor& dest,
            float alpha,
            const tensor& src
        );

        void add (
            tensor& dest,
            const tensor& src1,
            const tensor& src2
        );

    // -----------------------------------------------------------------------------------

        void affine_transform_conv(
            tensor& dest,
            const tensor& src,
            const tensor& A,
            const tensor& B
        );
        
    // -----------------------------------------------------------------------------------

        void softmax (
            tensor& dest,
            const tensor& src
        );

    // ------------------------------------------------------------------------------------

        void sigmoid (
            tensor& dest,
            const tensor& src
        );

    // ------------------------------------------------------------------------------------

        void relu (
            tensor& dest,
            const tensor& src
        );

    // ------------------------------------------------------------------------------------

        void tanh (
            tensor& dest,
            const tensor& src
        );

    // -----------------------------------------------------------------------------------

        class tensor_conv
        {
        public:
            tensor_conv(const tensor_conv&) = delete;
            tensor_conv& operator=(const tensor_conv&) = delete;

            tensor_conv() {}

            void clear(
            ) {}

            void setup(
                const tensor& data,    /* not used but required for interface */
                const tensor& filters, /* not used but required for interface */
                int stride_y,
                int stride_x,
                int padding_y,
                int padding_x
            ) 
            {
                (void)data;    /* silence compiler */
                DLIB_CASSERT(stride_y > 0 && stride_x > 0);
                DLIB_CASSERT(0 <= padding_y && padding_y < filters.nr());
                DLIB_CASSERT(0 <= padding_x && padding_x < filters.nc());
                last_stride_y = stride_y;
                last_stride_x = stride_x;
                last_padding_y = padding_y;
                last_padding_x = padding_x;            
            }

             void operator() (
                const bool add_to_output,
                tensor& output,
                const tensor& data,
                const tensor& filters
            );

        private:

            long last_stride_y = 0;
            long last_stride_x = 0;
            long last_padding_y = 0;
            long last_padding_x = 0;
        };

    // -----------------------------------------------------------------------------------

    } 
}

#ifdef NO_MAKEFILE
#include "cpu_dlib.cpp"
#endif

#endif // DLIB_DNN_CPU_H_


