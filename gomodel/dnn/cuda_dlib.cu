// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "cuda_utils.h"
#include "cuda_dlib.h"


namespace dlib 
{ 
    namespace cuda 
    {

    // -----------------------------------------------------------------------------------

        void set_device (
            int dev
        )
        {
            CHECK_CUDA(cudaSetDevice(dev));
        }

        int get_device (
        )
        {
            int dev = 0;
            CHECK_CUDA(cudaGetDevice(&dev));
            return dev;
        }

        std::string get_device_name (
            int device
        )
        {
            cudaDeviceProp props;
            CHECK_CUDA(cudaGetDeviceProperties(&props, device));
            return props.name;
        }

        void set_current_device_blocking_sync(
        )
        {
            CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
        }

        int get_num_devices (
        )
        {
            int num_devices;
            CHECK_CUDA(cudaGetDeviceCount(&num_devices));
            return num_devices;
        }

        bool can_access_peer (int device_id, int peer_device_id)
        {
            int can_access;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, device_id, peer_device_id));
            return can_access != 0;
        }
        bool can_access_peer (const tensor& device, const tensor& peer_device)
        {
            return can_access_peer(device.device_id(), peer_device.device_id());
        }

        void device_synchronize (int dev) 
        { 
            raii_set_device set_dev(dev);
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        void device_synchronize (const tensor& dev) { device_synchronize(dev.device_id()); }

        enable_peer_access::
        enable_peer_access(
            int device_id,
            int peer_device_id
        ) : call_disable(false), device_id(device_id), peer_device_id(peer_device_id)
        {
            raii_set_device set_dev(device_id);

            auto err = cudaDeviceEnablePeerAccess(peer_device_id, 0);
            if (err == cudaSuccess)
            {
                call_disable = true;
            }
            else if (err == cudaErrorPeerAccessAlreadyEnabled)
            {
                // call cudaGetLastError() to dispose of this error since we don't
                // care.
                auto err2 = cudaGetLastError();
                if (err2 != cudaErrorPeerAccessAlreadyEnabled)
                    CHECK_CUDA(err2);
            }
            else
            {
                CHECK_CUDA(err);
            }
        }


        enable_peer_access::
        ~enable_peer_access() noexcept(false)
        {
            if (call_disable)
            {
                raii_set_device set_dev(device_id);
                CHECK_CUDA(cudaDeviceDisablePeerAccess(peer_device_id));
            }
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_scale_columns(float* out, const float* m, const float* v, size_t nr, size_t nc)
        {
            for (auto j : grid_stride_range(0, nr*nc))
            {
                out[j] = m[j]*v[j%nc];
            }
        }

        void scale_columns (
            tensor& out,
            const tensor& m,
            const tensor& v
        )
        {
            launch_kernel(_cuda_scale_columns, max_jobs(m.size()), out.device(), m.device(), v.device(), m.num_samples(), m.size()/m.num_samples());
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_scale_rows(float* out, const float* m, const float* v, size_t nr, size_t nc)
        {
            for (auto j : grid_stride_range(0, nr*nc))
            {
                out[j] = m[j]*v[j/nc];
            }
        }

        void scale_rows (
            tensor& out,
            const tensor& m,
            const tensor& v
        )
        {
            launch_kernel(_cuda_scale_rows, max_jobs(m.size()), out.device(), m.device(), v.device(), m.num_samples(), m.size()/m.num_samples());
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_scale_rows2(float* out, const float* m1, const float* m2, const float* v1, const float* v2, size_t nr, size_t nc)
        {
            for (auto j : grid_stride_range(0, nr*nc))
            {
                out[j] = (m1[j] - m2[j]*v1[j/nc]) * v2[j/nc];
            }
        }

        __global__ void _cuda_scale_rows2_beta(const float beta, float* out, const float* m1, const float* m2, const float* v1, const float* v2, size_t nr, size_t nc)
        {
            for (auto j : grid_stride_range(0, nr*nc))
            {
                out[j] = beta*out[j] + (m1[j] - m2[j]*v1[j/nc]) * v2[j/nc];
            }
        }

        void scale_rows2 (
            float beta, 
            tensor& out,
            const tensor& m1,
            const tensor& m2,
            const tensor& v1,
            const tensor& v2
        )
        {
            if (beta == 0)
            {
                launch_kernel(_cuda_scale_rows2, max_jobs(m1.size()), out.device(),
                    m1.device(), m2.device(), v1.device(), v2.device(), m1.num_samples(),
                    m1.size()/m1.num_samples());
            }
            else
            {
                launch_kernel(_cuda_scale_rows2_beta, max_jobs(m1.size()), beta,
                    out.device(), m1.device(), m2.device(), v1.device(), v2.device(),
                    m1.num_samples(), m1.size()/m1.num_samples());
            }
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_exp(float* dest, const float* src, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
                dest[i] = ::exp(src[i]);
        }

        void exp (
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_ASSERT(dest.size() == src.size());
            launch_kernel(_cuda_exp, max_jobs(src.size()), dest.device(), src.device(), src.size());
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_log(float* dest, const float* src, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
                dest[i] = ::log(src[i]);
        }

        void log (
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_ASSERT(dest.size() == src.size());
            launch_kernel(_cuda_log, max_jobs(src.size()), dest.device(), src.device(), src.size());
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_log10(float* dest, const float* src, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
                dest[i] = ::log10(src[i]);
        }

        void log10 (
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_ASSERT(dest.size() == src.size());
            launch_kernel(_cuda_log10, max_jobs(src.size()), dest.device(), src.device(), src.size());
        }

    // -----------------------------------------------------------------------------------

        __global__ void _cuda_multiply1(float* d, const float* s1, const float* s2, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = s1[i]*s2[i];
            }
        }
        __global__ void _cuda_multiply2(float* d, const float* s1, const float* s2, 
                                       size_t n, size_t s1_n, size_t s2_n, size_t max_size)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = 0;
                for (size_t j = i; j < max_size; j += n)
                    d[i] += s1[j%s1_n]*s2[j%s2_n];
            }
        }

        __global__ void _cuda_multiply3(float* d, const float* s1, const float* s2, 
                                       size_t n, size_t s1_n, size_t s2_n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = s1[i%s1_n]*s2[i%s2_n];
            }
        }

        __global__ void _cuda_multiply1_add_to(float* d, const float* s1, const float* s2, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] += s1[i]*s2[i];
            }
        }
        __global__ void _cuda_multiply2_add_to(float* d, const float* s1, const float* s2, 
                                       size_t n, size_t s1_n, size_t s2_n, size_t max_size)
        {
            for (auto i : grid_stride_range(0, n))
            {
                for (size_t j = i; j < max_size; j += n)
                    d[i] += s1[j%s1_n]*s2[j%s2_n];
            }
        }

        __global__ void _cuda_multiply3_add_to(float* d, const float* s1, const float* s2, 
                                       size_t n, size_t s1_n, size_t s2_n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] += s1[i%s1_n]*s2[i%s2_n];
            }
        }

        void multiply (
            bool add_to,
            tensor& dest,
            const tensor& src1,
            const tensor& src2
        )
        {

            DLIB_CASSERT(dest.k() == src1.k() && src1.k() == src2.k() &&
                dest.nr() == src1.nr() && src1.nr() == src2.nr() &&
                dest.nc() == src1.nc() && src1.nc() == src2.nc() );
            const long MD = std::max(std::max(dest.num_samples(),src1.num_samples()),src2.num_samples());
            DLIB_CASSERT((dest.num_samples()==1 || dest.num_samples()==MD) &&
                (src1.num_samples()==1 || src1.num_samples()==MD) &&
                (src2.num_samples()==1 || src2.num_samples()==MD) );

            if (dest.size() == 0)
                return;

            const size_t max_size = std::max(std::max(dest.size(),src1.size()),src2.size());
            const auto d = dest.host();
            const auto s1 = src1.host();
            const auto s2 = src2.host();
            if (dest.size() == src1.size() && src1.size() == src2.size())
            {
                if (add_to)
                    launch_kernel(_cuda_multiply1_add_to,max_jobs(dest.size()),dest.device(), src1.device(), src2.device(), src1.size());
                else
                    launch_kernel(_cuda_multiply1,max_jobs(dest.size()),dest.device(), src1.device(), src2.device(), src1.size());
            }
            else if (dest.num_samples() == 1)
            {
                if (add_to)
                    launch_kernel(_cuda_multiply2_add_to,max_jobs(dest.size()),dest.device(), src1.device(), src2.device(), 
                                                dest.size(), src1.size(), src2.size(), max_size);
                else
                    launch_kernel(_cuda_multiply2,max_jobs(dest.size()),dest.device(), src1.device(), src2.device(), 
                                                dest.size(), src1.size(), src2.size(), max_size);
            }
            else
            {
                if (add_to)
                    launch_kernel(_cuda_multiply3_add_to,max_jobs(dest.size()),dest.device(), src1.device(), src2.device(), 
                                                dest.size(), src1.size(), src2.size());
                else
                    launch_kernel(_cuda_multiply3,max_jobs(dest.size()),dest.device(), src1.device(), src2.device(), 
                                                dest.size(), src1.size(), src2.size());
            }
        }

    // ------------------------------------------------------------------------------------

        __global__ void _cuda_mult1(float* d, const float* s1, const float* s2, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = s1[i]*s2[i];
            }
        }

        __global__ void _cuda_mult1_add_to(float* d, const float* s1, const float* s2, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] += s1[i]*s2[i];
            }
        }

        __global__ void _cuda_mult2(float* d, const float* s1, const float* s2, 
                                   size_t dn, size_t dk, size_t dr, size_t dc,
                                   size_t s1n, size_t s1k, size_t s1r, size_t s1c,
                                   size_t s2n, size_t s2k, size_t s2r, size_t s2c)
        {
            for (auto i : grid_stride_range(0, dn*dk*dr*dc))
            {
                size_t n,k,r,c;
                unpack_idx(i, dk,dr,dc, n,k,r,c);

                float v1 = 0;
                float v2 = 0;

                if (n < s1n &&
                    k < s1k &&
                    r < s1r &&
                    c < s1c )
                {
                    v1 = s1[pack_idx(s1k,s1r,s1c, n,k,r,c)];
                }

                if (n < s2n &&
                    k < s2k &&
                    r < s2r &&
                    c < s2c )
                {
                    v2 = s2[pack_idx(s2k,s2r,s2c, n,k,r,c)];
                }

                d[i] = v1*v2;
            }
        }

        __global__ void _cuda_mult2_add_to(float* d, const float* s1, const float* s2, 
                                   size_t dn, size_t dk, size_t dr, size_t dc,
                                   size_t s1n, size_t s1k, size_t s1r, size_t s1c,
                                   size_t s2n, size_t s2k, size_t s2r, size_t s2c)
        {
            for (auto i : grid_stride_range(0, dn*dk*dr*dc))
            {
                size_t n,k,r,c;
                unpack_idx(i, dk,dr,dc, n,k,r,c);

                float v1 = 0;
                float v2 = 0;

                if (n < s1n &&
                    k < s1k &&
                    r < s1r &&
                    c < s1c )
                {
                    v1 = s1[pack_idx(s1k,s1r,s1c, n,k,r,c)];
                }

                if (n < s2n &&
                    k < s2k &&
                    r < s2r &&
                    c < s2c )
                {
                    v2 = s2[pack_idx(s2k,s2r,s2c, n,k,r,c)];
                }

                d[i] += v1*v2;
            }
        }

        void multiply_zero_padded (
            bool add_to,
            tensor& dest,
            const tensor& src1,
            const tensor& src2
        )
        {
            if (dest.size() == 0)
                return;

            // Do the simple and fast version if everything has the same dimensions
            if (have_same_dimensions(dest, src1) &&
                have_same_dimensions(dest, src2))
            {
                if (add_to)
                    launch_kernel(_cuda_mult1_add_to,max_jobs(dest.size()), dest.device(), src1.device(), src2.device(), dest.size());
                else
                    launch_kernel(_cuda_mult1,max_jobs(dest.size()), dest.device(), src1.device(), src2.device(), dest.size());
            }
            else
            {
                if (add_to)
                {
                    // Otherwise, do the more complex version with bounds checking.
                    launch_kernel(_cuda_mult2_add_to,max_jobs(dest.size()),
                                dest.device(), src1.device(), src2.device(), 
                                dest.num_samples(), dest.k(), dest.nr(), dest.nc(),
                                src1.num_samples(), src1.k(), src1.nr(), src1.nc(),
                                src2.num_samples(), src2.k(), src2.nr(), src2.nc()
                                );
                }
                else
                {
                    // Otherwise, do the more complex version with bounds checking.
                    launch_kernel(_cuda_mult2,max_jobs(dest.size()),
                                dest.device(), src1.device(), src2.device(), 
                                dest.num_samples(), dest.k(), dest.nr(), dest.nc(),
                                src1.num_samples(), src1.k(), src1.nr(), src1.nc(),
                                src2.num_samples(), src2.k(), src2.nr(), src2.nc()
                                );
                }
            }
        }

    // ------------------------------------------------------------------------------------

        __global__ void _cuda_add1(float* d, const float* s1, const float* s2, size_t n)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = s1[i]+s2[i];
            }
        }

        __global__ void _cuda_add2(float* d, const float* s1, const float* s2, 
                                   size_t dn, size_t dk, size_t dr, size_t dc,
                                   size_t s1n, size_t s1k, size_t s1r, size_t s1c,
                                   size_t s2n, size_t s2k, size_t s2r, size_t s2c)
        {
            for (auto i : grid_stride_range(0, dn*dk*dr*dc))
            {
                size_t n,k,r,c;
                unpack_idx(i, dk,dr,dc, n,k,r,c);

                float v1 = 0;
                float v2 = 0;

                if (n < s1n &&
                    k < s1k &&
                    r < s1r &&
                    c < s1c )
                {
                    v1 = s1[pack_idx(s1k,s1r,s1c, n,k,r,c)];
                }

                if (n < s2n &&
                    k < s2k &&
                    r < s2r &&
                    c < s2c )
                {
                    v2 = s2[pack_idx(s2k,s2r,s2c, n,k,r,c)];
                }

                d[i] = v1+v2;
            }
        }

        void add (
            tensor& dest,
            const tensor& src1,
            const tensor& src2
        )
        {
            if (dest.size() == 0)
                return;

            // Do the simple and fast version if everything has the same dimensions
            if (have_same_dimensions(dest, src1) &&
                have_same_dimensions(dest, src2))
            {
                launch_kernel(_cuda_add1,max_jobs(dest.size()), dest.device(), src1.device(), src2.device(), dest.size());
            }
            else
            {
                // Otherwise, do the more complex version with bounds checking.
                launch_kernel(_cuda_add2,max_jobs(dest.size()),
                            dest.device(), src1.device(), src2.device(), 
                            dest.num_samples(), dest.k(), dest.nr(), dest.nc(),
                            src1.num_samples(), src1.k(), src1.nr(), src1.nc(),
                            src2.num_samples(), src2.k(), src2.nr(), src2.nc()
                            );
            }

        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_add_scaled(float* d, const float* s, size_t n, float scale)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] += scale*s[i]; 
            }
        }

        void add_scaled(
            tensor& dest,
            const float scale,
            const tensor& src
        )
        {
            DLIB_CASSERT(dest.size()==src.size());
            launch_kernel(_cuda_add_scaled,max_jobs(dest.size()),dest.device(), src.device(), dest.size(), scale);
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_add_cv_to_all_columns(float beta, float* dest, float alpha, const float* src, size_t size, size_t stride)
        {
            for (auto i : grid_stride_range(0, size))
            {
                dest[i] = beta*dest[i] + alpha*src[i/stride];
            }
        }

        __global__ void _cuda_add_cv_to_all_columns_no_beta(float* dest, float alpha, const float* src, size_t size, size_t stride)
        {
            for (auto i : grid_stride_range(0, size))
            {
                dest[i] = alpha*src[i/stride];
            }
        }

        void add_cv_to_all_columns(
            float beta, 
            tensor& dest, 
            float alpha, 
            const tensor& src
        )
        {
            DLIB_CASSERT(dest.num_samples() == src.num_samples() && src.num_samples() == src.size());
            if (beta == 0)
                launch_kernel(_cuda_add_cv_to_all_columns_no_beta, max_jobs(dest.size()), dest.device(), alpha, src.device(), dest.size(), dest.size()/dest.num_samples());
            else
                launch_kernel(_cuda_add_cv_to_all_columns, max_jobs(dest.size()), beta, dest.device(), alpha, src.device(), dest.size(), dest.size()/dest.num_samples());
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_affine_transform_range(
            float* d, const float* s1, const float* s2, const float* s3, size_t begin, size_t end, float A, float B, float C
        )
        {
            for (auto i : grid_stride_range(begin, end))
            {
                d[i] = A*s1[i] + B*s2[i] + C*s3[i];
            }
        }


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
        )
        {
            DLIB_CASSERT(dest.size()==src1.size());
            DLIB_CASSERT(dest.size()==src2.size());
            DLIB_CASSERT(dest.size()==src3.size());
            DLIB_CASSERT(begin <= end && end <= dest.size());
            launch_kernel(_cuda_affine_transform_range,max_jobs(end-begin),
                dest.device(), src1.device(),
                src2.device(), src3.device(), begin, end, A, B, C);
        }


    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_compute_adam_update(
            size_t begin,
            size_t end,
            float* s,
            float* m,
            float* v,
            const float alpha,
            const float weight_decay,
            const float momentum1,
            const float momentum2,
            const float* params,
            const float* params_grad
        )
        {
            const float eps = 1e-8;
            // The loop is equivalent to doing this:
            //   m = momentum1*m + (1-momentum1)    *   (weight_decay*params + params_grad);
            //   v = momentum2*v + (1-momentum2)*squared(weight_decay*params + params_grad);
            //   s = -alpha*m/(sqrt(v) + eps);
            for (auto i : grid_stride_range(begin, end))
            {
                float g = (weight_decay*params[i] + params_grad[i]);
                m[i] = momentum1*m[i] + (1-momentum1)*g;
                v[i] = momentum2*v[i] + (1-momentum2)*g*g;
                s[i] = -alpha*m[i]/(std::sqrt(v[i]) + eps);
            }
        }

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
        )
        {
            DLIB_CASSERT(s.size() == m.size() &&
                         s.size() == v.size() &&
                         s.size() == params.size() &&
                         s.size() == params_grad.size());
            DLIB_CASSERT(begin <= end && end <= params.size());
            const float alpha = learning_rate*std::sqrt(1-std::pow(momentum2,t))/(1-std::pow(momentum1, t));

            launch_kernel(_cuda_compute_adam_update,max_jobs(end-begin),
                    begin, end, s.device(), m.device(), v.device(), alpha, weight_decay,
                    momentum1, momentum2, params.device(), params_grad.device());
        }

    // -----------------------------------------------------------------------------------

        __global__ void _cuda_affine_transform_conv(float* d, const float* s, size_t n, const float* A, const float* B, size_t bs, size_t ks)
        {
            for (auto i : grid_stride_range(0, n))
            {
                auto k = (i/bs)%ks;
                d[i] = A[k]*s[i] + B[k];
            }
        }

        void affine_transform_conv(
            tensor& dest,
            const tensor& src,
            const tensor& A,
            const tensor& B
        )
        {
            DLIB_CASSERT(have_same_dimensions(dest, src));
            DLIB_CASSERT(have_same_dimensions(A, B));
            DLIB_CASSERT(A.num_samples() == 1 && A.nr() == 1 && A.nc() == 1 && A.k() == src.k());

            launch_kernel(_cuda_affine_transform_conv,max_jobs(dest.size()),
                    dest.device(), src.device(), src.size(), A.device(), B.device(), src.nr()*src.nc(), src.k());
        }


    // ----------------------------------------------------------------------------------------

        __global__ void _set_tensor(float* out, size_t n, const float val)
        {
            for (auto i : grid_stride_range(0, n))
                out[i] = val;
        }

        void set_tensor (
            tensor& t,
            float value
        )
        {
            launch_kernel(_set_tensor, max_jobs(t.size()), t.device(), t.size(), value);
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _scale_tensor(float* out, size_t n, const float val)
        {
            for (auto i : grid_stride_range(0, n))
                out[i] *= val;
        }

        void scale_tensor (
            tensor& t,
            float value
        )
        {
            launch_kernel(_scale_tensor, max_jobs(t.size()), t.device(), t.size(), value);
        }

    // -----------------------------------------------------------------------------------
    // -----------------------------------------------------------------------------------

        __global__ void _cuda_threshold(float* d, size_t n, float thresh)
        {
            for (auto i : grid_stride_range(0, n))
            {
                d[i] = d[i]>thresh ? 1:0;
            }
        }

        void threshold (
            tensor& data,
            float thresh
        )
        {
            launch_kernel(_cuda_threshold,max_jobs(data.size()),data.device(), data.size(), thresh);
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_prelu(const float* s, float* d, size_t n, const float* pp)
        {
            const float p = *pp;
            for (auto i : grid_stride_range(0, n))
            {
                if (s[i] > 0)
                    d[i] = s[i];
                else
                    d[i] = p*s[i];
            }
        }

        void prelu (
            tensor& dest,
            const tensor& src,
            const tensor& param
        )
        {
            launch_kernel(_cuda_prelu, max_jobs(dest.size()), 
                src.device(), dest.device(), src.size(), param.device());
        }

    // ----------------------------------------------------------------------------------------

        __global__ void _cuda_copy_tensor_add_to (float* dest, size_t size,  const float* src,  size_t dest_stride, size_t src_stride, size_t block_size)
        {
            for(auto i : grid_stride_range(0, size)) 
            {
                size_t blk = i/block_size;
                size_t j = i%block_size;
                dest[blk*dest_stride + j] += src[blk*src_stride + j];
            }
        }

        __global__ void _cuda_copy_tensor (float* dest, size_t size,  const float* src,  size_t dest_stride, size_t src_stride, size_t block_size)
        {
            for(auto i : grid_stride_range(0, size)) 
            {
                size_t blk = i/block_size;
                size_t j = i%block_size;
                dest[blk*dest_stride + j] = src[blk*src_stride + j];
            }
        }

        void copy_tensor(
            bool add_to,
            tensor& dest,
            size_t dest_k_offset,
            const tensor& src,
            size_t src_k_offset,
            size_t count_k
        )
        {
            const size_t dest_sample_size = static_cast<size_t>(dest.nc() * dest.nr() * dest.k());
            const size_t src_sample_size = static_cast<size_t>(src.nc() * src.nr() * src.k());

            const size_t block_size = count_k * dest.nc() * dest.nr();

            DLIB_CASSERT(dest.num_samples() == src.num_samples() &&
                         dest.nc() == src.nc() && dest.nr() == src.nr(), "All sources should fit into dest tensor size");
            DLIB_CASSERT(dest.k() - dest_k_offset >= count_k, "Not enough space in dest tensor");
            DLIB_CASSERT(src.k() - src_k_offset >= count_k, "Not enough space in src tensor");

            float* dest_p = dest.device() + dest_k_offset * dest.nc() * dest.nr();
            const float* src_p = src.device() + src_k_offset * src.nc() * src.nr();;

            if (add_to)
            {
                launch_kernel(_cuda_copy_tensor_add_to, max_jobs(dest.size()), 
                              dest_p, block_size*dest.num_samples(),
                              src_p, dest_sample_size, src_sample_size, block_size);
            }
            else
            {
                launch_kernel(_cuda_copy_tensor, max_jobs(dest.size()), 
                              dest_p, block_size*dest.num_samples(),
                              src_p, dest_sample_size, src_sample_size, block_size);
            }
        }

    // ----------------------------------------------------------------------------------------

    }
}

