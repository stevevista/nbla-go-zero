// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CPU_cPP_
#define DLIB_DNN_CPU_cPP_

// This file contains CPU implementations of the GPU based functions in cuda_dlib.h

#include "cpu_dlib.h"
#include "tensor_tools.h"
#include "../rectangle.h"

namespace dlib
{
    namespace cpu 
    {

    // -----------------------------------------------------------------------------------

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


            if (beta == 0 && alpha == 0)
            {
                dest = 0;
                return;
            }

            auto d = dest.host();
            auto s = src.host();
            for (long n = 0; n < dest.num_samples(); ++n)
            {
                const auto sn = src.num_samples()==1 ? 0:n;
                for (long k = 0; k < dest.k(); ++k)
                {
                    const auto sk = src.k()==1 ? 0:k;
                    for (long r = 0; r < dest.nr(); ++r)
                    {
                        const auto sr = src.nr()==1 ? 0:r;
                        for (long c = 0; c < dest.nc(); ++c)
                        {
                            const auto sc = src.nc()==1 ? 0:c;

                            const auto s_idx = ((sn*src.k() + sk)*src.nr() + sr)*src.nc() + sc;
                            *d = beta*(*d) + alpha*s[s_idx];
                            ++d;
                        }
                    }
                }
            }
        }

    // ----------------------------------------------------------------------------------------

        void add (
            tensor& dest,
            const tensor& src1,
            const tensor& src2
        )
        {
            auto d = dest.host();
            auto s1 = src1.host();
            auto s2 = src2.host();

            // Do the simple and fast version if everything has the same dimensions
            if (have_same_dimensions(dest, src1) &&
                have_same_dimensions(dest, src2))
            {
                for (size_t i = 0; i < dest.size(); ++i)
                    d[i] = s1[i] + s2[i];
                return;
            }

            // Otherwise, do the more complex version with bounds checking.
            for (long n = 0; n < dest.num_samples(); ++n)
            {
                for (long k = 0; k < dest.k(); ++k)
                {
                    for (long r = 0; r < dest.nr(); ++r)
                    {
                        for (long c = 0; c < dest.nc(); ++c)
                        {
                            float v1 = 0;
                            float v2 = 0;

                            // if this index is inside src1
                            if (n < src1.num_samples() && 
                                k < src1.k() && 
                                r < src1.nr() && 
                                c < src1.nc() )
                            {
                                const auto s_idx = ((n*src1.k() + k)*src1.nr() + r)*src1.nc() + c;
                                v1 = s1[s_idx];
                            }

                            // if this index is inside src2
                            if (n < src2.num_samples() && 
                                k < src2.k() && 
                                r < src2.nr() && 
                                c < src2.nc() )
                            {
                                const auto s_idx = ((n*src2.k() + k)*src2.nr() + r)*src2.nc() + c;
                                v2 = s2[s_idx];
                            }

                            *d = v1 + v2;
                            ++d;
                        }
                    }
                }
            }
        }

    // -----------------------------------------------------------------------------------

        void affine_transform_conv(
            tensor& dest,
            const tensor& src,
            const tensor& A,
            const tensor& B
        )
        {
            DLIB_CASSERT(have_same_dimensions(dest,src));
            DLIB_CASSERT(have_same_dimensions(A,B));
            DLIB_CASSERT(A.num_samples() == 1 &&
                         A.nr() == 1 &&
                         A.nc() == 1 &&
                         A.k() == src.k());

            auto d = dest.host();
            auto s = src.host();
            const auto a = A.host();
            const auto b = B.host();
            for (long n = 0; n < dest.num_samples(); ++n)
            {
                for (long k = 0; k < dest.k(); ++k)
                {
                    for (long r = 0; r < dest.nr(); ++r)
                    {
                        for (long c = 0; c < dest.nc(); ++c)
                        {
                            *d++ = a[k]*(*s++) + b[k];
                        }
                    }
                }
            }
        }

    // -----------------------------------------------------------------------------------
    // -----------------------------------------------------------------------------------
    // -----------------------------------------------------------------------------------

        namespace ttimpl
        {
        void softmax (
            const long num_locations,
            const long num_channels,
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_ASSERT(num_channels*num_locations == src.nr()*src.nc()*src.k());
            DLIB_CASSERT(have_same_dimensions(dest,src));
            const auto d = dest.host();
            const auto s = src.host();

            // Note that we subtract out the max values in each channel before applying
            // exp() to avoid numeric overflow in the subsequent computations.  Doing this
            // doesn't change the resulting output, it just makes it more numerically
            // stable.
            for (long n = 0; n < src.num_samples(); ++n)
            {
                auto ss = s + num_locations*num_channels*n;
                auto dd = d + num_locations*num_channels*n;
                for (long i = 0; i < num_locations; ++i)
                {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (long k = 0; k < num_channels; ++k)
                        max_val = std::max(max_val, ss[k*num_locations]);

                    for (long k = 0; k < num_channels; ++k)
                        dd[k*num_locations] = std::exp(ss[k*num_locations]-max_val);

                    ++ss;
                    ++dd;
                }
            }

            // Now normalize each channel so they sum to 1.
            for (long n = 0; n < src.num_samples(); ++n)
            {
                const auto dd = d + num_locations*num_channels*n;
                for (long i = 0; i < num_locations; ++i)
                {
                    const auto ddd = dd+i;

                    float temp = 0;
                    for (long k = 0; k < num_channels; ++k)
                        temp += ddd[k*num_locations];
                    for (long k = 0; k < num_channels; ++k)
                        ddd[k*num_locations] /= temp;
                }
            }
        }

        }

    // ----------------------------------------------------------------------------------------

        void softmax (
            tensor& dest,
            const tensor& src
        )
        {
            DLIB_CASSERT(have_same_dimensions(dest,src));
            ttimpl::softmax(src.nr()*src.nc(), src.k(), dest, src);
        }

    // ------------------------------------------------------------------------------------

        void sigmoid (
            tensor& dest,
            const tensor& src
        )
        {
            const auto d = dest.host();
            const auto s = src.host();
            for (size_t i = 0; i < src.size(); ++i)
                d[i] = 1/(1+std::exp(-s[i]));
        }

    // ------------------------------------------------------------------------------------

        void relu (
            tensor& dest,
            const tensor& src
        )
        {
            dest = lowerbound(mat(src), 0);
        }

    // ------------------------------------------------------------------------------------

        void tanh (
            tensor& dest,
            const tensor& src
        )
        {
            const auto d = dest.host();
            const auto s = src.host();
            for (size_t i = 0; i < src.size(); ++i)
                d[i] = std::tanh(s[i]);
        }

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

        void img2col(
            matrix<float>& output,
            const tensor& data,
            long n,
            long filter_nr,
            long filter_nc,
            long stride_y,
            long stride_x,
            long padding_y,
            long padding_x
        )
        {
            const auto d = data.host() + data.k()*data.nr()*data.nc()*n;
            const rectangle boundary = get_rect(data);

            const long out_nr = 1+(data.nr()+2*padding_y-filter_nr)/stride_y;
            const long out_nc = 1+(data.nc()+2*padding_x-filter_nc)/stride_x;

            output.set_size(out_nr*out_nc, 
                            data.k()*filter_nr*filter_nc);
            DLIB_CASSERT(output.size() != 0);
            float* t = &output(0,0);

            // now fill in the Toeplitz output matrix for the n-th sample in data.  
            size_t cnt = 0;
            const long max_r = data.nr() + padding_y-(filter_nr-1);
            const long max_c = data.nc() + padding_x-(filter_nc-1);
            for (long r = -padding_y; r < max_r; r+=stride_y)
            {
                for (long c = -padding_x; c < max_c; c+=stride_x)
                {
                    for (long k = 0; k < data.k(); ++k)
                    {
                        for (long y = 0; y < filter_nr; ++y)
                        {
                            for (long x = 0; x < filter_nc; ++x)
                            {
                                DLIB_ASSERT(cnt < output.size());
                                long xx = c+x;
                                long yy = r+y;
                                if (boundary.contains(xx,yy))
                                    *t = d[(k*data.nr() + yy)*data.nc() + xx];
                                else
                                    *t = 0;
                                ++t;
                                ++cnt;
                            }
                        }
                    }
                }
            }
        }

        void tensor_conv::operator() (
            const bool add_to_output,
            tensor& output,
            const tensor& data,
            const tensor& filters
        )
        {
            DLIB_CASSERT(last_stride_y > 0 && last_stride_x > 0, "You must call setup() before calling this function.");
            output.set_size(data.num_samples(),
                            filters.num_samples(),
                            1+(data.nr()+2*last_padding_y-filters.nr())/last_stride_y,
                            1+(data.nc()+2*last_padding_x-filters.nc())/last_stride_x);

            DLIB_CASSERT(is_same_object(output,data) == false);
            DLIB_CASSERT(is_same_object(output,filters) == false);
            DLIB_CASSERT(filters.k() == data.k());
            DLIB_CASSERT(last_stride_y > 0 && last_stride_x > 0, "You must call setup() before calling this function.");
            DLIB_CASSERT(filters.nr() <= data.nr() + 2*last_padding_y,
                "Filter windows must be small enough to fit into the padded image.");
            DLIB_CASSERT(filters.nc() <= data.nc() + 2*last_padding_x,
                "Filter windows must be small enough to fit into the padded image.");

            DLIB_CASSERT(output.num_samples() == data.num_samples());
            DLIB_CASSERT(output.k() == filters.num_samples());
            DLIB_CASSERT(output.nr() == 1+(data.nr()+2*last_padding_y-filters.nr())/last_stride_y);
            DLIB_CASSERT(output.nc() == 1+(data.nc()+2*last_padding_x-filters.nc())/last_stride_x);


            matrix<float> temp;
            for (long n = 0; n < data.num_samples(); ++n)
            {
                img2col(temp, data, n, filters.nr(), filters.nc(), last_stride_y, last_stride_x, last_padding_y, last_padding_x);

                if (add_to_output)
                    output.add_to_sample(n, mat(filters)*trans(temp));
                else 
                    output.set_sample(n, mat(filters)*trans(temp));
            }
        }

    } 
}


#endif // DLIB_DNN_CPU_cPP_


