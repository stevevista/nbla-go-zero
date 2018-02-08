// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_DEFAULT_MULTIPLY_
#define DLIB_MATRIx_DEFAULT_MULTIPLY_


#include "matrix.h"
#include "matrix_utilities.h"
#include "../enable_if.h"

namespace dlib
{


// ----------------------------------------------------------------------------------------
    
    class rectangle
    {
    public:

        rectangle (
            long l_,
            long t_,
            long r_,
            long b_
        ) :
            l(l_),
            t(t_),
            r(r_),
            b(b_)
        {}

        long top (
        ) const { return t; }


        long left (
        ) const { return l; }

        long right (
        ) const { return r; }

        long bottom (
        ) const { return b; }
       
        unsigned long width (
        ) const 
        { 
            if (is_empty())
                return 0;
            else
                return r - l + 1; 
        }

        unsigned long height (
        ) const 
        { 
            if (is_empty())
                return 0;
            else
                return b - t + 1; 
        }

        bool is_empty (
        ) const { return (t > b || l > r); }


        bool operator== (
            const rectangle& rect 
        ) const 
        {
            return (l == rect.l) && (t == rect.t) && (r == rect.r) && (b == rect.b);
        }

        bool operator!= (
            const rectangle& rect 
        ) const 
        {
            return !(*this == rect);
        }

        inline bool operator< (const dlib::rectangle& b) const
        { 
            if      (left() < b.left()) return true;
            else if (left() > b.left()) return false;
            else if (top() < b.top()) return true;
            else if (top() > b.top()) return false;
            else if (right() < b.right()) return true;
            else if (right() > b.right()) return false;
            else if (bottom() < b.bottom()) return true;
            else if (bottom() > b.bottom()) return false;
            else                    return false;
        }

    private:
        const long l;
        const long t;
        const long r;
        const long b;   
    };

// ------------------------------------------------------------------------------------

    namespace ma
    {
        template < typename EXP, typename enable = void >
        struct matrix_is_vector { static const bool value = false; };
        template < typename EXP >
        struct matrix_is_vector<EXP, typename enable_if_c<EXP::NR==1 || EXP::NC==1>::type > { static const bool value = true; };
    }

// ------------------------------------------------------------------------------------

    /*!  This file defines the default_matrix_multiply() function.  It is a function 
         that conforms to the following definition:

        template <
            typename matrix_dest_type,
            typename EXP1,
            typename EXP2
            >
        void default_matrix_multiply (
            matrix_dest_type& dest,
            const EXP1& lhs,
            const EXP2& rhs
        );
            requires
                - (lhs*rhs).destructively_aliases(dest) == false
                - dest.nr() == (lhs*rhs).nr()
                - dest.nc() == (lhs*rhs).nc()
            ensures
                - #dest == dest + lhs*rhs
    !*/

// ------------------------------------------------------------------------------------

    template <
        typename matrix_dest_type,
        typename EXP1,
        typename EXP2
        >
    typename enable_if_c<ma::matrix_is_vector<EXP1>::value == true || ma::matrix_is_vector<EXP2>::value == true>::type 
    default_matrix_multiply (
        matrix_dest_type& dest,
        const EXP1& lhs,
        const EXP2& rhs
    )
    {
        matrix_assign_default(dest, lhs*rhs, 1, true);
    }

// ------------------------------------------------------------------------------------

    template <
        typename matrix_dest_type,
        typename EXP1,
        typename EXP2
        >
    typename enable_if_c<ma::matrix_is_vector<EXP1>::value == false && ma::matrix_is_vector<EXP2>::value == false>::type 
    default_matrix_multiply (
        matrix_dest_type& dest,
        const EXP1& lhs,
        const EXP2& rhs
    )
    {
        const long bs = 90;

        // if the matrices are small enough then just use the simple multiply algorithm
        if (lhs.nc() <= 2 || rhs.nc() <= 2 || lhs.nr() <= 2 || rhs.nr() <= 2 || (lhs.size() <= bs*10 && rhs.size() <= bs*10) )
        {
            matrix_assign_default(dest, lhs*rhs, 1, true);
        }
        else
        {
            // if the lhs and rhs matrices are big enough we should use a cache friendly
            // algorithm that computes the matrix multiply in blocks.  


            // Loop over all the blocks in the lhs matrix
            for (long r = 0; r < lhs.nr(); r+=bs)
            {
                for (long c = 0; c < lhs.nc(); c+=bs)
                {
                    // make a rect for the block from lhs 
                    rectangle lhs_block(c, r, std::min(c+bs-1,lhs.nc()-1), std::min(r+bs-1,lhs.nr()-1));

                    // now loop over all the rhs blocks we have to multiply with the current lhs block
                    for (long i = 0; i < rhs.nc(); i += bs)
                    {
                        // make a rect for the block from rhs 
                        rectangle rhs_block(i, c, std::min(i+bs-1,rhs.nc()-1), std::min(c+bs-1,rhs.nr()-1));

                        // This loop is optimized assuming that the data is laid out in 
                        // row major order in memory.
                        for (long r = lhs_block.top(); r <= lhs_block.bottom(); ++r)
                        {
                            for (long c = lhs_block.left(); c<= lhs_block.right(); ++c)
                            {
                                const typename EXP2::type temp = lhs(r,c);
                                for (long i = rhs_block.left(); i <= rhs_block.right(); ++i)
                                {
                                    dest(r,i) += rhs(c,i)*temp;
                                }
                            }
                        }
                    }
                }
            }
        }


    }

// ------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_DEFAULT_MULTIPLY_

