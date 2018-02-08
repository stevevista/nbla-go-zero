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

    // -----------------------------------------------------------------------------------
    // -----------------------------------------------------------------------------------
    // -----------------------------------------------------------------------------------

        __global__ void _cuda_inverse_norms(float* invnorms, const float* data, size_t nr, size_t nc, const float eps)
        {
            // initialize invnorms before we begin.
            for (auto i : grid_stride_range_y(0, nr))
                for (auto j : grid_stride_range(0, 1))
                    invnorms[i] = eps;
            __syncthreads();

            for (auto i : grid_stride_range_y(0, nr))
            {
                auto p = data + i*nc;
                float temp = 0;
                for (auto j : grid_stride_range(0, nc))
                    temp += p[j]*p[j];

                // and store the sum into invnorms[i]
                warp_reduce_atomic_add(invnorms[i], temp);
            }
            __syncthreads();

            for (auto i : grid_stride_range_y(0, nr))
                for (auto j : grid_stride_range(0, 1))
                    invnorms[i] = 1.0/std::sqrt(invnorms[i]);
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

    }
}

