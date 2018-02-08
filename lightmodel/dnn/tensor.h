// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_TENSOR_H_
#define DLIB_DNn_TENSOR_H_

#include <cstring>
#include "../matrix.h"
#include "cudnn_dlibapi.h"
#include "gpu_data.h"
#include <memory>


namespace dlib
{

// ----------------------------------------------------------------------------------------

    class tensor;
    namespace cuda
    {
        void set_tensor (
            tensor& t,
            float value
        );

        void scale_tensor (
            tensor& t,
            float value
        );
    }

// ----------------------------------------------------------------------------------------

    class tensor
    {
    public:
        tensor(
        ) : m_n(0), m_k(0), m_nr(0), m_nc(0), m_size(0)
        {}

        template <typename EXP>
        tensor(
            const matrix_exp<EXP>& item
        )
        {
            set_size(item.nr(), item.nc());
            *this = item;
        }

        explicit tensor(
            long n_, long k_ = 1, long nr_ = 1, long nc_ = 1
        ) 
        {
            DLIB_ASSERT( n_ >= 0 && k_ >= 0 && nr_ >= 0 && nc_ >= 0);

            set_size(n_,k_,nr_,nc_);
        }

        tensor(const tensor& item)
        {
            copy_size(item);
            memcpy(*this, item);
        }

        tensor(tensor&& item) { swap(item); }
        tensor& operator=(tensor&& item) { swap(item); return *this; }


        long num_samples() const { return m_n; }
        long k() const { return m_k; }
        long nr() const { return m_nr; }
        long nc() const { return m_nc; }
        size_t size() const { return m_size; }

        typedef float* iterator;
        typedef const float* const_iterator;
        iterator       begin()       { return host(); }
        const_iterator begin() const { return host(); }
        iterator       end()         { return host()+size(); }
        const_iterator end() const   { return host()+size(); }

        void async_copy_to_device() const
        {
            data().async_copy_to_device();
        }

        const float* host() const { return data_instance.host(); }
        float*       host()       { return data_instance.host(); }
        float*       host_write_only() { return data_instance.host_write_only(); }
        const float* device() const { return data_instance.device(); }
        float*       device()       { return data_instance.device(); }
        float*       device_write_only() { return data_instance.device_write_only(); }


        int device_id() const { return data().device_id(); }

        tensor& operator= (float val)
        {
#ifdef DLIB_USE_CUDA
            // If you are using CUDA then presumably you will be mostly using tensors on
            // the GPU.  So unless you seem to be actively working with the host side's
            // data then we do this initialization on the device side since this avoids a
            // host to device transfer that would likely immediately follow.
            if (data().device_ready())
            {
                cuda::set_tensor(*this, val);
                return *this;
            }
#endif
            auto d = host_write_only();
            for (size_t i = 0; i < size(); ++i)
                d[i] = val;

            return *this;
        }

        tensor& operator*= (float val)
        {
#ifdef DLIB_USE_CUDA
            cuda::scale_tensor(*this, val);
            return *this;
#else
            for (auto& d : *this)
                d *= val;

            return *this;
#endif
        }
        
        tensor& operator/= (float val)
        {
            *this *= 1.0/val;
            return *this;
        }

        template <typename EXP>
        tensor& operator= (const matrix_exp<EXP>& item)
        {
            if (!(num_samples() == item.nr() && k()*nr()*nc() == item.nc()))
                set_size(item.nr(), item.nc());
            set_ptrm(host_write_only(), m_n, m_nr*m_nc*m_k) = item;
            return *this;
        }

        template <typename EXP>
        tensor& operator+= (const matrix_exp<EXP>& item)
        {
            DLIB_CASSERT(num_samples() == item.nr() &&
                         nr()*nc()*k() == item.nc());
            static_assert((is_same_type<float, typename EXP::type>::value == true),
                "To assign a matrix to a tensor the matrix must contain float values");
            set_ptrm(host(), m_n, m_nr*m_nc*m_k) += item;
            return *this;
        }

        template <typename EXP>
        tensor& operator-= (const matrix_exp<EXP>& item)
        {
            DLIB_CASSERT(num_samples() == item.nr() &&
                         nr()*nc()*k() == item.nc());
            static_assert((is_same_type<float, typename EXP::type>::value == true),
                "To assign a matrix to a tensor the matrix must contain float values");
            set_ptrm(host(), m_n, m_nr*m_nc*m_k) -= item;
            return *this;
        }

        template <typename EXP>
        void set_sample (
            unsigned long idx,
            const matrix_exp<EXP>& item
        )
        {
            DLIB_CASSERT(idx < (unsigned long)num_samples());
            DLIB_CASSERT(item.size() == nr()*nc()*k());
            static_assert((is_same_type<float, typename EXP::type>::value == true),
                "To assign a matrix to a tensor the matrix must contain float values");
            set_ptrm(host()+idx*item.size(), item.nr(), item.nc()) = item;
        }


        template <typename EXP>
        void add_to_sample (
            unsigned long idx,
            const matrix_exp<EXP>& item
        )
        {
            DLIB_CASSERT(idx < (unsigned long)num_samples());
            DLIB_CASSERT(item.size() == nr()*nc()*k());
            static_assert((is_same_type<float, typename EXP::type>::value == true),
                "To assign a matrix to a tensor the matrix must contain float values");
            set_ptrm(host()+idx*item.size(), item.nr(), item.nc()) += item;
        }

        friend void memcpy (
            tensor& dest, 
            const tensor& src
        )
        {
            DLIB_CASSERT(dest.size() == src.size());
            memcpy(dest.data(), 0,  
                   src.data(),  0, 
                   src.size());
        }

    public:
        

        void clear(
        )
        {
            set_size(0,0,0,0);
            // free underlying memory
            data_instance.set_size(0);
        }

        void copy_size (
            const tensor& item
        )
        {
            set_size(item.num_samples(), item.k(), item.nr(), item.nc());
        }


        void set_size(
            long n_, long k_ = 1, long nr_ = 1, long nc_ = 1
        )
        {
            DLIB_ASSERT( n_ >= 0 && k_ >= 0 && nr_ >= 0 && nc_ >= 0);

            m_n = n_;
            m_k = k_;
            m_nr = nr_;
            m_nc = nc_;
            m_size = n_*k_*nr_*nc_;
            if ((long)data_instance.size() < m_size)
                data_instance.set_size(m_size);
#ifdef DLIB_USE_CUDA
            cudnn_descriptor.set_size(m_n,m_k,m_nr,m_nc);
#endif
        }


        tensor& operator= (const tensor& item) 
        {
            tensor temp(item);
            temp.swap(*this);
            return *this;
        }

        void swap(tensor& item)
        {
            std::swap(m_n,    item.m_n);
            std::swap(m_k,    item.m_k);
            std::swap(m_nr,   item.m_nr);
            std::swap(m_nc,   item.m_nc);
            std::swap(m_size, item.m_size);
            std::swap(data_instance, item.data_instance);
#ifdef DLIB_USE_CUDA
            std::swap(cudnn_descriptor, item.cudnn_descriptor);
#endif
        }

#ifdef DLIB_USE_CUDA
        const cuda::tensor_descriptor& get_cudnn_tensor_descriptor (
        ) const { return cudnn_descriptor; }
#endif


    protected:

        long m_n;
        long m_k;
        long m_nr;
        long m_nc;
        long m_size; // always equal to m_n*m_k*m_nr*m_nc

    protected:

#ifdef DLIB_USE_CUDA
        cuda::tensor_descriptor cudnn_descriptor;
#endif 

        gpu_data data_instance;

        gpu_data& data() { return data_instance; }
        const gpu_data& data() const { return data_instance; }
    };

// ----------------------------------------------------------------------------------------

    inline const matrix_op<op_pointer_to_mat<float> > mat (
        const tensor& t,
        long nr,
        long nc
    )
    {
        DLIB_ASSERT(nr >= 0 && nc >= 0 , 
                    "\tconst matrix_exp mat(tensor, nr, nc)"
                    << "\n\t nr and nc must be >= 0"
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
        );
        DLIB_ASSERT(nr*nc == (long)t.size() , 
                    "\tconst matrix_exp mat(tensor, nr, nc)"
                    << "\n\t The sizes don't match up."
                    << "\n\t nr*nc:    " << nr*nc
                    << "\n\t t.size(): " << t.size()
        );
        typedef op_pointer_to_mat<float> op;
        return matrix_op<op>(op(t.host(),nr,nc));
    }

    inline const matrix_op<op_pointer_to_mat<float> > mat (
        const tensor& t
    )
    {
        if (t.size() != 0)
            return mat(t, t.num_samples(), t.size()/t.num_samples());
        else
            return mat((float*)0,0,0);
    }

    inline const matrix_op<op_pointer_to_mat<float> > image_plane (
        const tensor& t,
        long sample = 0,
        long k = 0
    )
    {
        DLIB_ASSERT(0 <= sample && sample < t.num_samples() &&
                    0 <= k && k < t.k() &&
                    t.size() != 0, 
                    "\tconst matrix_exp image_plane(tensor,sample,k)"
                    << "\n\t Invalid arguments were given to this function."
                    << "\n\t sample: " << sample
                    << "\n\t k:      " << k 
                    << "\n\t t.num_samples(): " << t.num_samples() 
                    << "\n\t t.k():           " << t.k() 
                    << "\n\t t.size():        " << t.size() 
        );


        typedef op_pointer_to_mat<float> op;
        return matrix_op<op>(op(t.host() + ((sample*t.k() + k)*t.nr())*t.nc(), 
                                t.nr(), 
                                t.nc()));
    }

// ----------------------------------------------------------------------------------------

    inline bool have_same_dimensions (
        const tensor& a,
        const tensor& b
    )
    {
        return a.num_samples() == b.num_samples() &&
               a.k()  == b.k() &&
               a.nr() == b.nr() &&
               a.nc() == b.nc();
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_TENSOR_H_

