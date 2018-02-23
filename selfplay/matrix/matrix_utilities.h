// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_UTILITIES_
#define DLIB_MATRIx_UTILITIES_

#include "matrix.h"
#include <cmath>
#include <complex>
#include <limits>
#include "../pixel.h"
#include <vector>
#include <algorithm>
#include "matrix_op.h"
#include "matrix_mat.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//            Helper templates for making operators used by expression objects
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <typename T>
    class matrix_range_exp;

    template <typename T>
    struct matrix_traits<matrix_range_exp<T> >
    {
        typedef T type;
        typedef const T const_ret_type;
        typedef default_memory_manager mem_manager_type;
        typedef row_major_layout layout_type;
        const static long NR = 1;
        const static long NC = 0;
        const static long cost = 1;
    };

    template <typename T>
    class matrix_range_exp : public matrix_exp<matrix_range_exp<T> >
    {
    public:
        typedef typename matrix_traits<matrix_range_exp>::type type;
        typedef typename matrix_traits<matrix_range_exp>::const_ret_type const_ret_type;
        typedef typename matrix_traits<matrix_range_exp>::mem_manager_type mem_manager_type;
        const static long NR = matrix_traits<matrix_range_exp>::NR;
        const static long NC = matrix_traits<matrix_range_exp>::NC;
        const static long cost = matrix_traits<matrix_range_exp>::cost;
        typedef typename matrix_traits<matrix_range_exp>::layout_type layout_type;


        matrix_range_exp (
            T start_,
            T end_
        ) 
        {
            start = start_;
            if (start_ <= end_)
                inc = 1;
            else 
                inc = -1;
            nc_ = std::abs(end_ - start_) + 1;
        }
        matrix_range_exp (
            T start_,
            T inc_,
            T end_
        ) 
        {
            start = start_;
            nc_ = std::abs(end_ - start_)/inc_ + 1;
            if (start_ <= end_)
                inc = inc_;
            else
                inc = -inc_;
        }

        matrix_range_exp (
            T start_,
            T end_,
            long num,
            bool
        ) 
        {
            start = start_;
            nc_ = num;
            if (num > 1)
            {
                inc = (end_-start_)/(num-1);
            }
            else 
            {
                inc = 0;
                start = end_;
            }

        }

        const_ret_type operator() (
            long, 
            long c
        ) const { return start + c*inc;  }

        const_ret_type operator() (
            long c
        ) const { return start + c*inc;  }

        template <typename U>
        bool aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        template <typename U>
        bool destructively_aliases (
            const matrix_exp<U>& 
        ) const { return false; }

        long nr (
        ) const { return NR; }

        long nc (
        ) const { return nc_; }

        long nc_;
        T start;
        T inc;
    };

// ----------------------------------------------------------------------------------------

    template <typename EXP>
    inline bool is_row_vector (
        const matrix_exp<EXP>& m
    ) { return m.nr() == 1; }

    template <typename EXP>
    inline bool is_col_vector (
        const matrix_exp<EXP>& m
    ) { return m.nc() == 1; }

    template <typename EXP>
    inline bool is_vector (
        const matrix_exp<EXP>& m
    ) { return is_row_vector(m) || is_col_vector(m); }

// ----------------------------------------------------------------------------------------

    template <typename EXP>
    inline bool is_finite (
        const matrix_exp<EXP>& m
    ) 
    { 
        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                if (!is_finite(m(r,c)))
                    return false;
            }
        }
        return true;
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        template <typename T>
        const T& magnitude (const T& item) { return item; }
        template <typename T>
        T magnitude (const std::complex<T>& item) { return std::norm(item); }
    }

    template <
        typename EXP
        >
    void find_min_and_max (
        const matrix_exp<EXP>& m,
        typename EXP::type& min_val,
        typename EXP::type& max_val
    )
    {
        DLIB_ASSERT(m.size() > 0, 
            "\ttype find_min_and_max(const matrix_exp& m, min_val, max_val)"
            << "\n\tYou can't ask for the min and max of an empty matrix"
            << "\n\tm.size():     " << m.size() 
            );
        typedef typename matrix_exp<EXP>::type type;

        min_val = m(0,0);
        max_val = min_val;
        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                type temp = m(r,c);
                if (dlib::impl::magnitude(temp) > dlib::impl::magnitude(max_val))
                    max_val = temp;
                if (dlib::impl::magnitude(temp) < dlib::impl::magnitude(min_val))
                    min_val = temp;
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    point max_point (
        const matrix_exp<EXP>& m
    )
    {
        DLIB_ASSERT(m.size() > 0, 
            "\tpoint max_point(const matrix_exp& m)"
            << "\n\tm can't be empty"
            << "\n\tm.size():   " << m.size() 
            << "\n\tm.nr():     " << m.nr() 
            << "\n\tm.nc():     " << m.nc() 
            );
        typedef typename matrix_exp<EXP>::type type;

        point best_point(0,0);
        type val = m(0,0);
        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                type temp = m(r,c);
                if (dlib::impl::magnitude(temp) > dlib::impl::magnitude(val))
                {
                    val = temp;
                    best_point = point(c,r);
                }
            }
        }
        return best_point;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    point min_point (
        const matrix_exp<EXP>& m
    )
    {
        DLIB_ASSERT(m.size() > 0, 
            "\tpoint min_point(const matrix_exp& m)"
            << "\n\tm can't be empty"
            << "\n\tm.size():   " << m.size() 
            << "\n\tm.nr():     " << m.nr() 
            << "\n\tm.nc():     " << m.nc() 
            );
        typedef typename matrix_exp<EXP>::type type;

        point best_point(0,0);
        type val = m(0,0);
        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                type temp = m(r,c);
                if (dlib::impl::magnitude(temp) < dlib::impl::magnitude(val))
                {
                    val = temp;
                    best_point = point(c,r);
                }
            }
        }
        return best_point;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    long index_of_max (
        const matrix_exp<EXP>& m
    )
    {
        DLIB_ASSERT(m.size() > 0 && is_vector(m) == true, 
            "\tlong index_of_max(const matrix_exp& m)"
            << "\n\tm must be a row or column matrix"
            << "\n\tm.size():   " << m.size() 
            << "\n\tm.nr():     " << m.nr() 
            << "\n\tm.nc():     " << m.nc() 
            );
        typedef typename matrix_exp<EXP>::type type;

        type val = m(0);
        long best_idx = 0;
        for (long i = 1; i < m.size(); ++i)
        {
            type temp = m(i);
            if (dlib::impl::magnitude(temp) > dlib::impl::magnitude(val))
            {
                val = temp;
                best_idx = i;
            }
        }
        return best_idx;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    long index_of_min (
        const matrix_exp<EXP>& m
    )
    {
        DLIB_ASSERT(m.size() > 0 && is_vector(m), 
            "\tlong index_of_min(const matrix_exp& m)"
            << "\n\tm must be a row or column matrix"
            << "\n\tm.size():   " << m.size() 
            << "\n\tm.nr():     " << m.nr() 
            << "\n\tm.nc():     " << m.nc() 
            );
        typedef typename matrix_exp<EXP>::type type;

        type val = m(0);
        long best_idx = 0;
        for (long i = 1; i < m.size(); ++i)
        {
            type temp = m(i);
            if (dlib::impl::magnitude(temp) < dlib::impl::magnitude(val))
            {
                val = temp;
                best_idx = i;
            }
        }
        return best_idx;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    const typename matrix_exp<EXP>::type max (
        const matrix_exp<EXP>& m
    )
    {
        DLIB_ASSERT(m.size() > 0, 
            "\ttype max(const matrix_exp& m)"
            << "\n\tYou can't ask for the max() of an empty matrix"
            << "\n\tm.size():     " << m.size() 
            );
        typedef typename matrix_exp<EXP>::type type;

        type val = m(0,0);
        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                type temp = m(r,c);
                if (dlib::impl::magnitude(temp) > dlib::impl::magnitude(val))
                    val = temp;
            }
        }
        return val;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    const typename matrix_exp<EXP>::type min (
        const matrix_exp<EXP>& m
    )
    {
        DLIB_ASSERT(m.size() > 0, 
            "\ttype min(const matrix_exp& m)"
            << "\n\tYou can't ask for the min() of an empty matrix"
            << "\n\tm.size():     " << m.size() 
            );
        typedef typename matrix_exp<EXP>::type type;

        type val = m(0,0);
        for (long r = 0; r < m.nr(); ++r)
        {
            for (long c = 0; c < m.nc(); ++c)
            {
                type temp = m(r,c);
                if (dlib::impl::magnitude(temp) < dlib::impl::magnitude(val))
                    val = temp;
            }
        }
        return val;
    }

// ----------------------------------------------------------------------------------------

    template <typename M1, typename M2>
    struct op_binary_min : basic_op_mm<M1,M2>
    {
        op_binary_min( const M1& m1_, const M2& m2_) : basic_op_mm<M1,M2>(m1_,m2_){}

        typedef typename M1::type type;
        typedef const type const_ret_type;
        const static long cost = M1::cost + M2::cost + 1;

        const_ret_type apply ( long r, long c) const
        { return std::min(this->m1(r,c),this->m2(r,c)); }
    };

    template <
        typename EXP1,
        typename EXP2
        >
    inline const matrix_op<op_binary_min<EXP1,EXP2> > min_pointwise (
        const matrix_exp<EXP1>& a,
        const matrix_exp<EXP2>& b 
    )
    {
        COMPILE_TIME_ASSERT((is_same_type<typename EXP1::type,typename EXP2::type>::value == true));
        COMPILE_TIME_ASSERT(EXP1::NR == EXP2::NR || EXP1::NR == 0 || EXP2::NR == 0);
        COMPILE_TIME_ASSERT(EXP1::NC == EXP2::NC || EXP1::NC == 0 || EXP2::NC == 0);
        DLIB_ASSERT(a.nr() == b.nr() &&
               a.nc() == b.nc(), 
            "\t const matrix_exp min_pointwise(const matrix_exp& a, const matrix_exp& b)"
            << "\n\ta.nr(): " << a.nr()
            << "\n\ta.nc(): " << a.nc() 
            << "\n\tb.nr(): " << b.nr()
            << "\n\tb.nc(): " << b.nc() 
            );
        typedef op_binary_min<EXP1,EXP2> op;
        return matrix_op<op>(op(a.ref(),b.ref()));
    }

// ----------------------------------------------------------------------------------------

    template <typename M1, typename M2, typename M3>
    struct op_min_pointwise3 : basic_op_mmm<M1,M2,M3>
    {
        op_min_pointwise3( const M1& m1_, const M2& m2_, const M3& m3_) : 
            basic_op_mmm<M1,M2,M3>(m1_,m2_,m3_){}

        typedef typename M1::type type;
        typedef const typename M1::type const_ret_type;
        const static long cost = M1::cost + M2::cost + M3::cost + 2;

        const_ret_type apply (long r, long c) const
        { return std::min(this->m1(r,c),std::min(this->m2(r,c),this->m3(r,c))); }
    };

    template <
        typename EXP1,
        typename EXP2,
        typename EXP3
        >
    inline const matrix_op<op_min_pointwise3<EXP1,EXP2,EXP3> > 
    min_pointwise (
        const matrix_exp<EXP1>& a,
        const matrix_exp<EXP2>& b, 
        const matrix_exp<EXP3>& c
    )
    {
        COMPILE_TIME_ASSERT((is_same_type<typename EXP1::type,typename EXP2::type>::value == true));
        COMPILE_TIME_ASSERT((is_same_type<typename EXP2::type,typename EXP3::type>::value == true));
        COMPILE_TIME_ASSERT(EXP1::NR == EXP2::NR || EXP1::NR == 0 || EXP2::NR == 0);
        COMPILE_TIME_ASSERT(EXP1::NC == EXP2::NC || EXP1::NR == 0 || EXP2::NC == 0);
        COMPILE_TIME_ASSERT(EXP2::NR == EXP3::NR || EXP2::NR == 0 || EXP3::NR == 0);
        COMPILE_TIME_ASSERT(EXP2::NC == EXP3::NC || EXP2::NC == 0 || EXP3::NC == 0);
        DLIB_ASSERT(a.nr() == b.nr() &&
               a.nc() == b.nc() &&
               b.nr() == c.nr() &&
               b.nc() == c.nc(), 
            "\tconst matrix_exp min_pointwise(a,b,c)"
            << "\n\tYou can only make a do a pointwise min between equally sized matrices"
            << "\n\ta.nr(): " << a.nr()
            << "\n\ta.nc(): " << a.nc() 
            << "\n\tb.nr(): " << b.nr()
            << "\n\tb.nc(): " << b.nc() 
            << "\n\tc.nr(): " << c.nr()
            << "\n\tc.nc(): " << c.nc() 
            );

        typedef op_min_pointwise3<EXP1,EXP2,EXP3> op;
        return matrix_op<op>(op(a.ref(),b.ref(),c.ref()));
    }

// ----------------------------------------------------------------------------------------

    template <typename M1, typename M2>
    struct op_binary_max : basic_op_mm<M1,M2>
    {
        op_binary_max( const M1& m1_, const M2& m2_) : basic_op_mm<M1,M2>(m1_,m2_){}

        typedef typename M1::type type;
        typedef const type const_ret_type;
        const static long cost = M1::cost + M2::cost + 1;

        const_ret_type apply ( long r, long c) const
        { return std::max(this->m1(r,c),this->m2(r,c)); }
    };

    template <
        typename EXP1,
        typename EXP2
        >
    inline const matrix_op<op_binary_max<EXP1,EXP2> > max_pointwise (
        const matrix_exp<EXP1>& a,
        const matrix_exp<EXP2>& b 
    )
    {
        COMPILE_TIME_ASSERT((is_same_type<typename EXP1::type,typename EXP2::type>::value == true));
        COMPILE_TIME_ASSERT(EXP1::NR == EXP2::NR || EXP1::NR == 0 || EXP2::NR == 0);
        COMPILE_TIME_ASSERT(EXP1::NC == EXP2::NC || EXP1::NC == 0 || EXP2::NC == 0);
        DLIB_ASSERT(a.nr() == b.nr() &&
               a.nc() == b.nc(), 
            "\t const matrix_exp max_pointwise(const matrix_exp& a, const matrix_exp& b)"
            << "\n\ta.nr(): " << a.nr()
            << "\n\ta.nc(): " << a.nc() 
            << "\n\tb.nr(): " << b.nr()
            << "\n\tb.nc(): " << b.nc() 
            );
        typedef op_binary_max<EXP1,EXP2> op;
        return matrix_op<op>(op(a.ref(),b.ref()));
    }

// ----------------------------------------------------------------------------------------

    template <typename M1, typename M2, typename M3>
    struct op_max_pointwise3 : basic_op_mmm<M1,M2,M3>
    {
        op_max_pointwise3( const M1& m1_, const M2& m2_, const M3& m3_) : 
            basic_op_mmm<M1,M2,M3>(m1_,m2_,m3_){}

        typedef typename M1::type type;
        typedef const typename M1::type const_ret_type;
        const static long cost = M1::cost + M2::cost + M3::cost + 2;

        const_ret_type apply (long r, long c) const
        { return std::max(this->m1(r,c),std::max(this->m2(r,c),this->m3(r,c))); }
    };

    template <
        typename EXP1,
        typename EXP2,
        typename EXP3
        >
    inline const matrix_op<op_max_pointwise3<EXP1,EXP2,EXP3> > 
    max_pointwise (
        const matrix_exp<EXP1>& a,
        const matrix_exp<EXP2>& b, 
        const matrix_exp<EXP3>& c
    )
    {
        COMPILE_TIME_ASSERT((is_same_type<typename EXP1::type,typename EXP2::type>::value == true));
        COMPILE_TIME_ASSERT((is_same_type<typename EXP2::type,typename EXP3::type>::value == true));
        COMPILE_TIME_ASSERT(EXP1::NR == EXP2::NR || EXP1::NR == 0 || EXP2::NR == 0);
        COMPILE_TIME_ASSERT(EXP1::NC == EXP2::NC || EXP1::NR == 0 || EXP2::NC == 0);
        COMPILE_TIME_ASSERT(EXP2::NR == EXP3::NR || EXP2::NR == 0 || EXP3::NR == 0);
        COMPILE_TIME_ASSERT(EXP2::NC == EXP3::NC || EXP2::NC == 0 || EXP3::NC == 0);
        DLIB_ASSERT(a.nr() == b.nr() &&
               a.nc() == b.nc() &&
               b.nr() == c.nr() &&
               b.nc() == c.nc(), 
            "\tconst matrix_exp max_pointwise(a,b,c)"
            << "\n\tYou can only make a do a pointwise max between equally sized matrices"
            << "\n\ta.nr(): " << a.nr()
            << "\n\ta.nc(): " << a.nc() 
            << "\n\tb.nr(): " << b.nr()
            << "\n\tb.nc(): " << b.nc() 
            << "\n\tc.nr(): " << c.nr()
            << "\n\tc.nc(): " << c.nc() 
            );

        typedef op_max_pointwise3<EXP1,EXP2,EXP3> op;
        return matrix_op<op>(op(a.ref(),b.ref(),c.ref()));
    }

// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    typename enable_if_c<std::numeric_limits<typename EXP::type>::is_integer, double>::type length (
        const matrix_exp<EXP>& m
    )
    {
        DLIB_ASSERT(is_vector(m) == true, 
            "\ttype length(const matrix_exp& m)"
            << "\n\tm must be a row or column vector"
            << "\n\tm.nr():     " << m.nr() 
            << "\n\tm.nc():     " << m.nc() 
            );
        
        return std::sqrt(static_cast<double>(sum(squared(m))));
    }
    
    template <
        typename EXP
        >
    typename disable_if_c<std::numeric_limits<typename EXP::type>::is_integer, const typename EXP::type>::type length (
        const matrix_exp<EXP>& m
    )
    {
        DLIB_ASSERT(is_vector(m) == true, 
            "\ttype length(const matrix_exp& m)"
            << "\n\tm must be a row or column vector"
            << "\n\tm.nr():     " << m.nr() 
            << "\n\tm.nc():     " << m.nc() 
            );
        return std::sqrt(sum(squared(m)));
    }
 
// ----------------------------------------------------------------------------------------

    template <
        typename EXP
        >
    const typename matrix_exp<EXP>::type length_squared (
        const matrix_exp<EXP>& m
    )
    {
        DLIB_ASSERT(is_vector(m) == true, 
            "\ttype length_squared(const matrix_exp& m)"
            << "\n\tm must be a row or column vector"
            << "\n\tm.nr():     " << m.nr() 
            << "\n\tm.nc():     " << m.nc() 
            );
        return sum(squared(m));
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    
    template <typename M>
    struct op_trans 
    {
        op_trans( const M& m_) : m(m_){}

        const M& m;

        const static long cost = M::cost;
        const static long NR = M::NC;
        const static long NC = M::NR;
        typedef typename M::type type;
        typedef typename M::const_ret_type const_ret_type;
        typedef typename M::mem_manager_type mem_manager_type;
        typedef typename M::layout_type layout_type;

        const_ret_type apply (long r, long c) const { return m(c,r); }

        long nr () const { return m.nc(); }
        long nc () const { return m.nr(); }

        template <typename U> bool aliases               ( const matrix_exp<U>& item) const { return m.aliases(item); }
        template <typename U> bool destructively_aliases ( const matrix_exp<U>& item) const { return m.aliases(item); }

    }; 

    template <
        typename M
        >
    const matrix_op<op_trans<M> > trans (
        const matrix_exp<M>& m
    )
    {
        typedef op_trans<M> op;
        return matrix_op<op>(op(m.ref()));
    }

// ----------------------------------------------------------------------------------------

    template <typename M, typename target_type>
    struct op_cast 
    {
        op_cast( const M& m_) : m(m_){}
        const M& m;

        const static long cost = M::cost+2;
        const static long NR = M::NR;
        const static long NC = M::NC;
        typedef target_type type;
        typedef const target_type const_ret_type;
        typedef typename M::mem_manager_type mem_manager_type;
        typedef typename M::layout_type layout_type;
        const_ret_type apply ( long r, long c) const { return static_cast<target_type>(m(r,c)); }

        long nr () const { return m.nr(); }
        long nc () const { return m.nc(); }

        template <typename U> bool aliases               ( const matrix_exp<U>& item) const { return m.aliases(item); }
        template <typename U> bool destructively_aliases ( const matrix_exp<U>& item) const { return m.destructively_aliases(item); }
    };

    template <
        typename target_type,
        typename EXP
        >
    const matrix_op<op_cast<EXP, target_type> > matrix_cast (
        const matrix_exp<EXP>& m
    )
    {
        typedef op_cast<EXP, target_type> op;
        return matrix_op<op>(op(m.ref()));
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        template <typename type, typename S>
        inline type greaterthan_eq(const type& val, const S& s)
        {
            if (val >= s)
                return 1;
            else
                return 0;
        }
        
    }
    DLIB_DEFINE_OP_MS(op_greaterthan_eq, impl::greaterthan_eq, 1);

    template <
        typename EXP,
        typename S
        >
    const typename enable_if<is_built_in_scalar_type<S>, matrix_op<op_greaterthan_eq<EXP,S> > >::type operator>= (
        const matrix_exp<EXP>& m,
        const S& s
    )
    {
        // you can only use this relational operator with the built in scalar types like
        // long, float, etc.
        COMPILE_TIME_ASSERT(is_built_in_scalar_type<typename EXP::type>::value);

        typedef op_greaterthan_eq<EXP,S> op;
        return matrix_op<op>(op(m.ref(),s));
    }

    template <
        typename EXP,
        typename S
        >
    const typename enable_if<is_built_in_scalar_type<S>, matrix_op<op_greaterthan_eq<EXP,S> > >::type operator<= (
        const S& s,
        const matrix_exp<EXP>& m
    )
    {
        // you can only use this relational operator with the built in scalar types like
        // long, float, etc.
        COMPILE_TIME_ASSERT(is_built_in_scalar_type<typename EXP::type>::value);

        typedef op_greaterthan_eq<EXP,S> op;
        return matrix_op<op>(op(m.ref(),s));
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        template <typename type, typename S>
        inline type equal_to(const type& val, const S& s)
        {
            if (val == s)
                return 1;
            else
                return 0;
        }
        
    }
    DLIB_DEFINE_OP_MS(op_equal_to, impl::equal_to, 1);

    template <
        typename EXP,
        typename S
        >
    const typename enable_if<is_built_in_scalar_type<S>, matrix_op<op_equal_to<EXP,S> > >::type operator== (
        const matrix_exp<EXP>& m,
        const S& s
    )
    {
        // you can only use this relational operator with the built in scalar types like
        // long, float, etc.
        COMPILE_TIME_ASSERT( is_built_in_scalar_type<typename EXP::type>::value);

        typedef op_equal_to<EXP,S> op;
        return matrix_op<op>(op(m.ref(),s));
    }

    template <
        typename EXP,
        typename S
        >
    const typename enable_if<is_built_in_scalar_type<S>, matrix_op<op_equal_to<EXP,S> > >::type operator== (
        const S& s,
        const matrix_exp<EXP>& m
    )
    {
        // you can only use this relational operator with the built in scalar types like
        // long, float, etc.
        COMPILE_TIME_ASSERT( is_built_in_scalar_type<typename EXP::type>::value);

        typedef op_equal_to<EXP,S> op;
        return matrix_op<op>(op(m.ref(),s));
    }

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        template <typename type, typename S>
        inline type not_equal_to(const type& val, const S& s)
        {
            if (val != s)
                return 1;
            else
                return 0;
        }
        
    }
    DLIB_DEFINE_OP_MS(op_not_equal_to, impl::not_equal_to, 1);


    template <
        typename EXP,
        typename S
        >
    const typename enable_if<is_built_in_scalar_type<S>, matrix_op<op_not_equal_to<EXP,S> > >::type operator!= (
        const matrix_exp<EXP>& m,
        const S& s
    )
    {
        // you can only use this relational operator with the built in scalar types like
        // long, float, etc.
        COMPILE_TIME_ASSERT(is_built_in_scalar_type<typename EXP::type>::value);

        typedef op_not_equal_to<EXP,S> op;
        return matrix_op<op>(op(m.ref(),s));
    }

    template <
        typename EXP,
        typename S
        >
    const typename enable_if<is_built_in_scalar_type<S>, matrix_op<op_not_equal_to<EXP,S> > >::type operator!= (
        const S& s,
        const matrix_exp<EXP>& m
    )
    {
        // you can only use this relational operator with the built in scalar types like
        // long, float, etc.
        COMPILE_TIME_ASSERT(is_built_in_scalar_type<typename EXP::type>::value);

        typedef op_not_equal_to<EXP,S> op;
        return matrix_op<op>(op(m.ref(),s));
    }

// ----------------------------------------------------------------------------------------

    template <
        typename P,
        int type = static_switch<
            pixel_traits<P>::grayscale,
            pixel_traits<P>::rgb,
            pixel_traits<P>::hsi,
            pixel_traits<P>::rgb_alpha,
            pixel_traits<P>::lab
            >::value
        >
    struct pixel_to_vector_helper;

    template <typename P>
    struct pixel_to_vector_helper<P,1>
    {
        template <typename M>
        static void assign (
            M& m,
            const P& pixel
        )
        {
            m(0) = static_cast<typename M::type>(pixel);
        }
    };

    template <typename P>
    struct pixel_to_vector_helper<P,2>
    {
        template <typename M>
        static void assign (
            M& m,
            const P& pixel
        )
        {
            m(0) = static_cast<typename M::type>(pixel.red);
            m(1) = static_cast<typename M::type>(pixel.green);
            m(2) = static_cast<typename M::type>(pixel.blue);
        }
    };

    template <typename P>
    struct pixel_to_vector_helper<P,3>
    {
        template <typename M>
        static void assign (
            M& m,
            const P& pixel
        )
        {
            m(0) = static_cast<typename M::type>(pixel.h);
            m(1) = static_cast<typename M::type>(pixel.s);
            m(2) = static_cast<typename M::type>(pixel.i);
        }
    };

    template <typename P>
    struct pixel_to_vector_helper<P,4>
    {
        template <typename M>
        static void assign (
            M& m,
            const P& pixel
        )
        {
            m(0) = static_cast<typename M::type>(pixel.red);
            m(1) = static_cast<typename M::type>(pixel.green);
            m(2) = static_cast<typename M::type>(pixel.blue);
            m(3) = static_cast<typename M::type>(pixel.alpha);
        }
    };

    template <typename P>
    struct pixel_to_vector_helper<P,5>
    {
        template <typename M>
        static void assign (
                M& m,
                const P& pixel
        )
        {
            m(0) = static_cast<typename M::type>(pixel.l);
            m(1) = static_cast<typename M::type>(pixel.a);
            m(2) = static_cast<typename M::type>(pixel.b);
        }
    };


    template <
        typename T,
        typename P
        >
    inline const matrix<T,pixel_traits<P>::num,1> pixel_to_vector (
        const P& pixel
    )
    {
        COMPILE_TIME_ASSERT(pixel_traits<P>::num > 0);
        matrix<T,pixel_traits<P>::num,1> m;
        pixel_to_vector_helper<P>::assign(m,pixel);
        return m;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename P,
        int type = static_switch<
            pixel_traits<P>::grayscale,
            pixel_traits<P>::rgb,
            pixel_traits<P>::hsi,
            pixel_traits<P>::rgb_alpha,
            pixel_traits<P>::lab
            >::value
        >
    struct vector_to_pixel_helper;

    template <typename P>
    struct vector_to_pixel_helper<P,1>
    {
        template <typename M>
        static void assign (
            P& pixel,
            const M& m
        )
        {
            pixel = static_cast<unsigned char>(m(0));
        }
    };

    template <typename P>
    struct vector_to_pixel_helper<P,2>
    {
        template <typename M>
        static void assign (
            P& pixel,
            const M& m
        )
        {
            pixel.red = static_cast<unsigned char>(m(0));
            pixel.green = static_cast<unsigned char>(m(1));
            pixel.blue = static_cast<unsigned char>(m(2));
        }
    };

    template <typename P>
    struct vector_to_pixel_helper<P,3>
    {
        template <typename M>
        static void assign (
            P& pixel,
            const M& m
        )
        {
            pixel.h = static_cast<unsigned char>(m(0));
            pixel.s = static_cast<unsigned char>(m(1));
            pixel.i = static_cast<unsigned char>(m(2));
        }
    };

    template <typename P>
    struct vector_to_pixel_helper<P,4>
    {
        template <typename M>
        static void assign (
            P& pixel,
            const M& m
        )
        {
            pixel.red = static_cast<unsigned char>(m(0));
            pixel.green = static_cast<unsigned char>(m(1));
            pixel.blue = static_cast<unsigned char>(m(2));
            pixel.alpha = static_cast<unsigned char>(m(3));
        }
    };

    template <typename P>
    struct vector_to_pixel_helper<P,5>
    {
        template <typename M>
        static void assign (
                P& pixel,
                const M& m
        )
        {
            pixel.l = static_cast<unsigned char>(m(0));
            pixel.a = static_cast<unsigned char>(m(1));
            pixel.b = static_cast<unsigned char>(m(2));
        }
    };

    template <
        typename P,
        typename EXP
        >
    inline void vector_to_pixel (
        P& pixel,
        const matrix_exp<EXP>& vector 
    )
    {
        COMPILE_TIME_ASSERT(pixel_traits<P>::num == matrix_exp<EXP>::NR);
        COMPILE_TIME_ASSERT(matrix_exp<EXP>::NC == 1);
        vector_to_pixel_helper<P>::assign(pixel,vector);
    }

// ----------------------------------------------------------------------------------------

    inline const matrix_range_exp<double> linspace (
        double start,
        double end,
        long num
    ) 
    { 
        DLIB_ASSERT(num >= 0, 
            "\tconst matrix_exp linspace(start, end, num)"
            << "\n\tInvalid inputs to this function"
            << "\n\tstart: " << start 
            << "\n\tend:   " << end
            << "\n\tnum:   " << num 
            );

        return matrix_range_exp<double>(start,end,num,false); 
    }

}

#endif // DLIB_MATRIx_UTILITIES_

