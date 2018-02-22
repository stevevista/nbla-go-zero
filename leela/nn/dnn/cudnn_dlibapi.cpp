// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuDNN_CPP_
#define DLIB_DNN_CuDNN_CPP_

#ifdef DLIB_USE_CUDA

#include "cudnn_dlibapi.h"
#include "tensor.h"
#include <cudnn.h>
#include <iostream>
#include <string>
#include <vector>
#include "cuda_utils.h"
#include "cpu_dlib.h"
#include "cuda_dlib.h"
#include "tensor_tools.h"

static const char* cudnn_get_error_string(cudnnStatus_t s)
{
    switch(s)
    {
        case CUDNN_STATUS_NOT_INITIALIZED: 
            return "CUDA Runtime API initialization failed.";
        case CUDNN_STATUS_ALLOC_FAILED: 
            return "CUDA Resources could not be allocated.";
        case CUDNN_STATUS_BAD_PARAM:
            return "CUDNN_STATUS_BAD_PARAM";
        case CUDNN_STATUS_EXECUTION_FAILED:
            return "CUDNN_STATUS_EXECUTION_FAILED";
        case CUDNN_STATUS_NOT_SUPPORTED:
            return "CUDNN_STATUS_NOT_SUPPORTED";
        case CUDNN_STATUS_ARCH_MISMATCH:
            return "CUDNN_STATUS_ARCH_MISMATCH: Your GPU is too old and not supported by cuDNN";
        default:
            return "A call to cuDNN failed";
    }
}

// Check the return value of a call to the cuDNN runtime for an error condition.
#define CHECK_CUDNN(call)                                                      \
do{                                                                              \
    const cudnnStatus_t error = call;                                         \
    if (error != CUDNN_STATUS_SUCCESS)                                        \
    {                                                                          \
        std::ostringstream sout;                                               \
        sout << "Error while calling " << #call << " in file " << __FILE__ << ":" << __LINE__ << ". ";\
        sout << "code: " << error << ", reason: " << cudnn_get_error_string(error);\
        throw dlib::cudnn_error(sout.str());                            \
    }                                                                          \
}while(false)


namespace dlib
{

    namespace cuda 
    {

    // ------------------------------------------------------------------------------------

        static cudnnTensorDescriptor_t descriptor(const tensor& t) 
        {
            return (const cudnnTensorDescriptor_t)t.get_cudnn_tensor_descriptor().get_handle();
        }
        static cudnnTensorDescriptor_t descriptor(const tensor_descriptor& t) 
        {
            return (const cudnnTensorDescriptor_t)t.get_handle();
        }

    // ------------------------------------------------------------------------------------

        class cudnn_context
        {
        public:
            // not copyable
            cudnn_context(const cudnn_context&) = delete;
            cudnn_context& operator=(const cudnn_context&) = delete;

            cudnn_context()
            {
                handles.resize(16);
            }
            ~cudnn_context()
            {
                for (auto h : handles)
                {
                    if (h)
                        cudnnDestroy(h);
                }
            }

            cudnnHandle_t get_handle (
            )  
            { 
                int new_device_id;
                CHECK_CUDA(cudaGetDevice(&new_device_id));
                // make room for more devices if needed
                if (new_device_id >= (long)handles.size())
                    handles.resize(new_device_id+16);

                // If we don't have a handle already for this device then make one
                if (!handles[new_device_id])
                    CHECK_CUDNN(cudnnCreate(&handles[new_device_id]));

                // Finally, return the handle for the current device
                return handles[new_device_id];
            }

        private:

            std::vector<cudnnHandle_t> handles;
        };

        static cudnnHandle_t context()
        {
            thread_local cudnn_context c;
            return c.get_handle();
        }
    // ------------------------------------------------------------------------------------

        class cudnn_device_buffer
        {
        public:
            // not copyable
            cudnn_device_buffer(const cudnn_device_buffer&) = delete;
            cudnn_device_buffer& operator=(const cudnn_device_buffer&) = delete;

            cudnn_device_buffer()
            {
                buffers.resize(16);
            }
            ~cudnn_device_buffer()
            {
            }

            std::shared_ptr<resizable_cuda_buffer> get_buffer (
            )
            {
                int new_device_id;
                CHECK_CUDA(cudaGetDevice(&new_device_id));
                // make room for more devices if needed
                if (new_device_id >= (long)buffers.size())
                    buffers.resize(new_device_id+16);

                // If we don't have a buffer already for this device then make one
                std::shared_ptr<resizable_cuda_buffer> buff = buffers[new_device_id].lock();
                if (!buff)
                {
                    buff = std::make_shared<resizable_cuda_buffer>();
                    buffers[new_device_id] = buff;
                }

                // Finally, return the buffer for the current device
                return buff;
            }

        private:

            std::vector<std::weak_ptr<resizable_cuda_buffer>> buffers;
        };


        static std::shared_ptr<resizable_cuda_buffer> device_global_buffer()
        {
            thread_local cudnn_device_buffer buffer;
            return buffer.get_buffer();
        }
    // ------------------------------------------------------------------------------------

        class cudnn_activation_descriptor
        {
        public:
            // not copyable 
            cudnn_activation_descriptor(const cudnn_activation_descriptor&) = delete;
            cudnn_activation_descriptor& operator=(const cudnn_activation_descriptor&) = delete;

            cudnn_activation_descriptor(
                cudnnActivationMode_t mode,
                cudnnNanPropagation_t reluNanOpt,
                double reluCeiling
            )
            {
                CHECK_CUDNN(cudnnCreateActivationDescriptor(&handle));
                CHECK_CUDNN(cudnnSetActivationDescriptor(handle, mode, reluNanOpt, reluCeiling));
            }

            ~cudnn_activation_descriptor()
            {
                cudnnDestroyActivationDescriptor(handle);
            }

            cudnnActivationDescriptor_t get_handle (
            ) 
            { 
                return handle; 
            }
        private:
            cudnnActivationDescriptor_t handle;
        };

        static cudnnActivationDescriptor_t relu_activation_descriptor()
        {
            thread_local cudnn_activation_descriptor des(CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN,0);
            return des.get_handle();
        }

        static cudnnActivationDescriptor_t sigmoid_activation_descriptor()
        {
            thread_local cudnn_activation_descriptor des(CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN,0);
            return des.get_handle();
        }

        static cudnnActivationDescriptor_t tanh_activation_descriptor()
        {
            thread_local cudnn_activation_descriptor des(CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN,0);
            return des.get_handle();
        }

    // ------------------------------------------------------------------------------------

        tensor_descriptor::
        tensor_descriptor(
        ) : handle(nullptr)
        {
        }

        tensor_descriptor::
        ~tensor_descriptor()
        {
            set_size(0,0,0,0);
        }

        void tensor_descriptor::
        set_size(
            int n, 
            int k,
            int nr, 
            int nc 
        )
        {
            if (handle)
            {
                cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)handle);
                handle = nullptr;
            }

            if (n != 0 && nr != 0 && nc != 0 && k != 0)
            {
                cudnnTensorDescriptor_t h;
                CHECK_CUDNN(cudnnCreateTensorDescriptor(&h));
                handle = h;

                CHECK_CUDNN(cudnnSetTensor4dDescriptor((cudnnTensorDescriptor_t)handle,
                        CUDNN_TENSOR_NCHW,
                        CUDNN_DATA_FLOAT,
                        n,
                        k,
                        nr,
                        nc));
            }
        }

        void tensor_descriptor::
        get_size (
            int& n, 
            int& k,
            int& nr,
            int& nc
        ) const
        {
            if (handle)
            {
                int nStride, cStride, hStride, wStride;
                cudnnDataType_t datatype;
                CHECK_CUDNN(cudnnGetTensor4dDescriptor((cudnnTensorDescriptor_t)handle,
                        &datatype,
                        &n,
                        &k,
                        &nr,
                        &nc,
                        &nStride,
                        &cStride,
                        &hStride,
                        &wStride));
            }
            else
            {
                n = 0;
                k = 0;
                nr = 0;
                nc = 0;
            }
        }

    // ------------------------------------------------------------------------------------

        void add(
            float beta,
            tensor& dest,
            float alpha,
            const tensor& src
        )
        {
            DLIB_CASSERT(
                  (have_same_dimensions(src, dest) ||
                  (src.num_samples()==1 && src.k()==dest.k() && src.nr()==1 && src.nc()==1) ||
                  (src.num_samples()==1 && src.k()==dest.k() && src.nr()==dest.nr() && src.nc()==dest.nc()) ||
                  (src.num_samples()==1 && src.k()==1 && src.nr()==dest.nr() && src.nc()==dest.nc()) ||
                  (src.num_samples()==dest.num_samples() && src.k()==1 && src.nr()==1 && src.nc()==1)) &&
                  is_same_object(src,dest) == false , 
                    "\n\t dest.num_samples(): " << dest.num_samples()
                    <<"\n\t dest.k():           " << dest.k()
                    <<"\n\t dest.nr():          " << dest.nr()
                    <<"\n\t dest.nc():          " << dest.nc()
                    <<"\n\t src.num_samples():  " << src.num_samples()
                    <<"\n\t src.k():            " << src.k()
                    <<"\n\t src.nr():           " << src.nr()
                    <<"\n\t src.nc():           " << src.nc()
                    );

            if (dest.size() == src.size() && beta == 1)
            {
                // Call the dlib function in this case since it's faster than the one that
                // comes with cuDNN (at least as of cuDNN v4).
                add_scaled(dest, alpha, src);
                return;
            }
            else if (src.num_samples()==dest.num_samples() && src.k()==1 && src.nr()==1 && src.nc()==1)
            {
                add_cv_to_all_columns(beta, dest, alpha, src);
                return;
            }

            CHECK_CUDNN(cudnnAddTensor(context(),
                                    &alpha,
                                    descriptor(src),
                                    src.device(),
                                    &beta,
                                    descriptor(dest),
                                    dest.device()));
        }

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

        tensor_conv::
        tensor_conv(
        ) : 
            filter_handle(nullptr),
            conv_handle(nullptr),
            forward_algo(0)
        {
            clear();
        }

        void tensor_conv::
        clear (
        )
        {
            if (filter_handle) 
                cudnnDestroyFilterDescriptor((cudnnFilterDescriptor_t)filter_handle);
            if (conv_handle) 
                cudnnDestroyConvolutionDescriptor((cudnnConvolutionDescriptor_t)conv_handle);
            filter_handle = nullptr;
            conv_handle = nullptr;
            out_num_samples = 0;
            out_k = 0;
            out_nr = 0;
            out_nc = 0;

            stride_y = 0;
            stride_x = 0;
            padding_y = 0;
            padding_x = 0;
            data_num_samples = 0;
            data_k = 0;
            data_nr = 0;
            data_nc = 0;
            filters_num_samples = 0;
            filters_k = 0;
            filters_nr = 0;
            filters_nc = 0;

            forward_algo = 0;
            forward_workspace_size_in_bytes = 0;
            forward_workspace.reset();
            workspace.reset();
        }

        void tensor_conv::
        setup(
            const tensor& data,
            const tensor& filters,
            int stride_y_,
            int stride_x_,
            int padding_y_,
            int padding_x_
        ) 
        {
            DLIB_CASSERT(data.k() == filters.k());

            // if the last call to setup gave the same exact settings then don't do
            // anything.
            if (stride_y_ == stride_y && 
                stride_x_ == stride_x &&
                padding_y_ == padding_y && 
                padding_x_ == padding_x &&
                data_num_samples == data.num_samples() &&
                data_k == data.k() &&
                data_nr == data.nr() &&
                data_nc == data.nc() &&
                filters_num_samples == filters.num_samples() &&
                filters_k == filters.k() &&
                filters_nr == filters.nr() &&
                filters_nc == filters.nc())
            {
                return;
            }

            clear();
            try
            {
                stride_y = stride_y_;
                stride_x = stride_x_;
                padding_y = padding_y_;
                padding_x = padding_x_;
                data_num_samples = data.num_samples();
                data_k = data.k();
                data_nr = data.nr();
                data_nc = data.nc();
                filters_num_samples = filters.num_samples();
                filters_k = filters.k();
                filters_nr = filters.nr();
                filters_nc = filters.nc();

                CHECK_CUDNN(cudnnCreateFilterDescriptor((cudnnFilterDescriptor_t*)&filter_handle));
                CHECK_CUDNN(cudnnSetFilter4dDescriptor((cudnnFilterDescriptor_t)filter_handle, 
                                                 CUDNN_DATA_FLOAT, 
                                                 CUDNN_TENSOR_NCHW,
                                                 filters.num_samples(),
                                                 filters.k(),
                                                 filters.nr(),
                                                 filters.nc()));

                CHECK_CUDNN(cudnnCreateConvolutionDescriptor((cudnnConvolutionDescriptor_t*)&conv_handle));
#if CUDNN_MAJOR >= 6
                CHECK_CUDNN(cudnnSetConvolution2dDescriptor((cudnnConvolutionDescriptor_t)conv_handle,
                        padding_y, // vertical padding
                        padding_x, // horizontal padding
                        stride_y,
                        stride_x,
                        1, 1, // must be 1,1
                        CUDNN_CROSS_CORRELATION,
                        CUDNN_DATA_FLOAT)); // could also be CUDNN_CONVOLUTION
#else
                CHECK_CUDNN(cudnnSetConvolution2dDescriptor((cudnnConvolutionDescriptor_t)conv_handle,
                        padding_y, // vertical padding
                        padding_x, // horizontal padding
                        stride_y,
                        stride_x,
                        1, 1, // must be 1,1
                        CUDNN_CROSS_CORRELATION)); // could also be CUDNN_CONVOLUTION
#endif

                CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
                        (const cudnnConvolutionDescriptor_t)conv_handle,
                        descriptor(data),
                        (const cudnnFilterDescriptor_t)filter_handle,
                        &out_num_samples,
                        &out_k,
                        &out_nr,
                        &out_nc));

                tensor_descriptor dest_desc;
                dest_desc.set_size(out_num_samples,out_k,out_nr,out_nc);

                // Pick which forward algorithm we will use and allocate the necessary
                // workspace buffer.
                cudnnConvolutionFwdAlgo_t forward_best_algo;
                CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(
                        context(), 
                        descriptor(data),
                        (const cudnnFilterDescriptor_t)filter_handle,
                        (const cudnnConvolutionDescriptor_t)conv_handle,
                        descriptor(dest_desc),
                        dnn_prefer_fastest_algorithms()?CUDNN_CONVOLUTION_FWD_PREFER_FASTEST:CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
                        std::numeric_limits<size_t>::max(),
                        &forward_best_algo));
                forward_algo = forward_best_algo;
                CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize( 
                        context(),
                        descriptor(data),
                        (const cudnnFilterDescriptor_t)filter_handle,
                        (const cudnnConvolutionDescriptor_t)conv_handle,
                        descriptor(dest_desc),
                        forward_best_algo,
                        &forward_workspace_size_in_bytes));

                workspace = device_global_buffer();
            }
            catch(...)
            {
                clear();
                throw;
            }
        }

        tensor_conv::
        ~tensor_conv (
        )
        {
            clear();
        }

        void tensor_conv::operator() (
            const bool add_to_output,
            resizable_tensor& output,
            const tensor& data,
            const tensor& filters
        )
        {
            DLIB_CASSERT(stride_y > 0 && stride_x > 0, "You must call setup() before calling this function");

            output.set_size(out_num_samples, out_k, out_nr, out_nc);
            (*this)(add_to_output, static_cast<tensor&>(output), data, filters);
        }

        void tensor_conv::operator() (
            const bool add_to_output,
            tensor& output,
            const tensor& data,
            const tensor& filters
        )
        {
            DLIB_CASSERT(is_same_object(output,data) == false);
            DLIB_CASSERT(is_same_object(output,filters) == false);
            DLIB_CASSERT(filters.k() == data.k());
            DLIB_CASSERT(stride_y > 0 && stride_x > 0, "You must call setup() before calling this function");
            DLIB_CASSERT(filters.nc() <= data.nc() + 2*padding_x,
                "Filter windows must be small enough to fit into the padded image."
                << "\n\t filters.nc(): " << filters.nc() 
                << "\n\t data.nc():  " << data.nc() 
                << "\n\t padding_x: " << padding_x 
                );
            DLIB_CASSERT(filters.nr() <= data.nr() + 2*padding_y,
                "Filter windows must be small enough to fit into the padded image."
                << "\n\t filters.nr(): " << filters.nr() 
                << "\n\t data.nr():  " << data.nr() 
                << "\n\t padding_y: " << padding_y 
                );


            DLIB_CASSERT(output.num_samples() == data.num_samples(),out_num_samples << "  " << data.num_samples());
            DLIB_CASSERT(output.k() == filters.num_samples());
            DLIB_CASSERT(output.nr() == 1+(data.nr()+2*padding_y-filters.nr())/stride_y);
            DLIB_CASSERT(output.nc() == 1+(data.nc()+2*padding_x-filters.nc())/stride_x);



            const float alpha = 1;
            const float beta = add_to_output ? 1 : 0;

            // Since cudnnConvolutionForward() is an asynchronous call, we need to hold a
            // reference to the workspace buffer so we can be sure it isn't reallocated
            // while the function is still executing on the device.  But each time we come
            // here, we make sure to grab the latest workspace buffer so that, globally, we
            // minimize the number of such buffers.
            forward_workspace = workspace->get(forward_workspace_size_in_bytes);

            CHECK_CUDNN(cudnnConvolutionForward(
                    context(),
                    &alpha,
                    descriptor(data),
                    data.device(),
                    (const cudnnFilterDescriptor_t)filter_handle,
                    filters.device(),
                    (const cudnnConvolutionDescriptor_t)conv_handle,
                    (cudnnConvolutionFwdAlgo_t)forward_algo,
                    forward_workspace,
                    forward_workspace_size_in_bytes,
                    &beta,
                    descriptor(output),
                    output.device()));
        }

           // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

    pooling::pooling (
    ) : handle(nullptr),window_height(0),window_width(0),stride_y(0),stride_x(0),padding_y(0), padding_x(0)
    {
    }

    pooling::~pooling(
    )
    {
        clear();
    }

    void pooling::
    clear(
    )
    {
        if (handle)
            cudnnDestroyPoolingDescriptor((cudnnPoolingDescriptor_t)handle);
        handle = nullptr;
        window_height = 0;
        window_width = 0;
        stride_y = 0;
        stride_x = 0;
        padding_y = 0;
        padding_x = 0;
    }

    void pooling::
    setup_max_pooling(
        int window_height_,
        int window_width_,
        int stride_y_,
        int stride_x_,
        int padding_y_,
        int padding_x_ 
    )
    {
        setup(window_height_, window_width_, stride_y_, stride_x_, padding_y_, padding_x_, CUDNN_POOLING_MAX);
        do_max_pooling = true;
    }

    void pooling::
    setup_avg_pooling(
        int window_height_,
        int window_width_,
        int stride_y_,
        int stride_x_,
        int padding_y_,
        int padding_x_
    )
    {
        setup(window_height_, window_width_, stride_y_, stride_x_, padding_y_, padding_x_, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);
        do_max_pooling = false;
    }

    void pooling::
    setup(
        int window_height_,
        int window_width_,
        int stride_y_,
        int stride_x_,
        int padding_y_,
        int padding_x_,
        int pooling_mode
    )
    {
        DLIB_CASSERT (window_height_ > 0 && window_width_ > 0 && 
                      stride_y_ > 0 && stride_x_ > 0 , 
                      "window_height_: " << window_height_ 
                      << "\t\n window_width_: " << window_width_ 
                      << "\t\n stride_y_: " << stride_y_ 
                      << "\t\n stride_x_: " << stride_x_ );
        DLIB_CASSERT( 0 <= padding_y_ && padding_y_ < window_height_ && 
                      0 <= padding_x_ && padding_x_ < window_width_,
                      "window_height_: " << window_height_ 
                      << "\t\n window_width_: " << window_width_ 
                      << "\t\n padding_y_: " << padding_y_ 
                      << "\t\n padding_x_: " << padding_x_ );

        if (window_height == window_height_ &&
            window_width  == window_width_ &&
            stride_y == stride_y_ &&
            stride_x == stride_x_ && 
            padding_y == padding_y_ &&
            padding_x == padding_x_
            )
        {
            return;
        }

        clear();
        try
        {
            window_height = window_height_;
            window_width = window_width_;
            stride_x = stride_x_;
            stride_y = stride_y_;
            padding_y  = padding_y_;
            padding_x  = padding_x_;
            cudnnPoolingDescriptor_t poolingDesc;
            CHECK_CUDNN(cudnnCreatePoolingDescriptor(&poolingDesc));
            handle = poolingDesc;

            CHECK_CUDNN(cudnnSetPooling2dDescriptor(poolingDesc,
                                            (cudnnPoolingMode_t)pooling_mode,
                                            CUDNN_PROPAGATE_NAN,
                                            window_height,
                                            window_width,
                                            padding_y,
                                            padding_x,  
                                            stride_y,
                                            stride_x));
        }
        catch(...)
        {
            clear();
            throw;
        }
    }

    void pooling::
    operator() (
        resizable_tensor& dest,
        const tensor& src
    )
    {
        DLIB_CASSERT(window_width  <= src.nc() + 2*padding_x,
            "Pooling windows must be small enough to fit into the padded image."
            << "\n\t window_width: " << window_width 
            << "\n\t src.nc():  " << src.nc() 
            << "\n\t padding_x: " << padding_x 
            );
        DLIB_CASSERT(window_height <= src.nr() + 2*padding_y,
            "Pooling windows must be small enough to fit into the padded image."
            << "\n\t window_height: " << window_height 
            << "\n\t src.nr():  " << src.nr() 
            << "\n\t padding_y: " << padding_y 
            );
        const float alpha = 1;
        const float beta = 0;
        int outN;
        int outC;
        int outH;
        int outW;
        CHECK_CUDNN(cudnnGetPooling2dForwardOutputDim((const cudnnPoolingDescriptor_t)handle,
                                                descriptor(src),
                                                &outN,
                                                &outC,
                                                &outH,
                                                &outW));


        dest.set_size(outN,outC,outH,outW);

        DLIB_CASSERT(dest.num_samples() == src.num_samples());
        DLIB_CASSERT(dest.k() == src.k());
        DLIB_CASSERT(dest.nr() == 1 + (src.nr() + 2*padding_y - window_height)/stride_y, 
            "\n stride_y:  " << stride_y  <<
            "\n padding_y: " << padding_y  <<
            "\n window_height: " << window_height  <<
            "\n src.nr(): " << src.nr()  <<
            "\n dest.nr(): " << dest.nr()  <<
            "\n src.nr()/stride_y: " <<  src.nr()/stride_y); 
        DLIB_CASSERT(dest.nc() == 1 + (src.nc() + 2*padding_x - window_width)/stride_x, 
            "\n stride_x:  " << stride_x  <<
            "\n padding_x: " << padding_x  <<
            "\n window_width: " << window_width  <<
            "\n src.nc(): " << src.nc()  <<
            "\n dest.nc(): " << dest.nc()  <<
            "\n src.nc()/stride_x: " <<  src.nc()/stride_x); 

        CHECK_CUDNN(cudnnPoolingForward(context(),
                                 (const cudnnPoolingDescriptor_t)handle,
                                 &alpha,
                                 descriptor(src),
                                 src.device(),
                                 &beta,
                                 descriptor(dest),
                                 dest.device()));
    }

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

        void softmax (
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_CASSERT(have_same_dimensions(dest,src));
            if (src.size() == 0)
                return;

            const float alpha = 1;
            const float beta = 0;

            CHECK_CUDNN(cudnnSoftmaxForward(context(),
                                      CUDNN_SOFTMAX_ACCURATE,
                                      CUDNN_SOFTMAX_MODE_CHANNEL,
                                      &alpha,
                                      descriptor(src),
                                      src.device(),
                                      &beta,
                                      descriptor(dest),
                                      dest.device()));
        }

    // ------------------------------------------------------------------------------------

        void relu (
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_CASSERT(have_same_dimensions(dest,src));
            if (src.size() == 0)
                return;

            const float alpha = 1;
            const float beta = 0;
            CHECK_CUDNN(cudnnActivationForward(context(),
                                         relu_activation_descriptor(),
                                         &alpha,
                                         descriptor(src),
                                         src.device(),
                                         &beta,
                                         descriptor(dest),
                                         dest.device()));
        }

    // ------------------------------------------------------------------------------------

        void tanh (
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_CASSERT(have_same_dimensions(dest,src));
            if (src.size() == 0)
                return;

            const float alpha = 1;
            const float beta = 0;
            CHECK_CUDNN(cudnnActivationForward(context(),
                                         tanh_activation_descriptor(),
                                         &alpha,
                                         descriptor(src),
                                         src.device(),
                                         &beta,
                                         descriptor(dest),
                                         dest.device()));
        }


    // ------------------------------------------------------------------------------------
    }
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuDNN_CPP_


