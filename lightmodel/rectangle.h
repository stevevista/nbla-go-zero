// Copyright (C) 2005  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_RECTANGLe_
#define DLIB_RECTANGLe_

#include "algs.h"
#include <algorithm>
#include <iostream>

namespace dlib
{


// ----------------------------------------------------------------------------------------

    template <typename image_type>
    inline long num_rows(
        const image_type& img
    ) { return img.nr(); }
    /*!
        ensures
            - By default, try to use the member function .nr() to determine the number
              of rows in an image.  However, as stated at the top of this file, image
              objects should provide their own overload of num_rows() if needed.
    !*/

    template <typename image_type>
    inline long num_columns(
        const image_type& img
    ) { return img.nc(); }

// ----------------------------------------------------------------------------------------
    
    class rectangle
    {
        /*!
            INITIAL VALUE
                The initial value of this object is defined by its constructor.

            CONVENTION
                left() == l
                top() == t
                right() == r
                bottom() == b
        !*/

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

        rectangle (
            unsigned long w,
            unsigned long h
        ) :
            l(0),
            t(0),
            r(static_cast<long>(w)-1),
            b(static_cast<long>(h)-1)
        {
            DLIB_ASSERT((w > 0 && h > 0) || (w == 0 && h == 0),
                "\trectangle(width,height)"
                << "\n\twidth and height must be > 0 or both == 0"
                << "\n\twidth:  " << w 
                << "\n\theight: " << h 
                << "\n\tthis: " << this
                );
        }

        rectangle (
        ) :
            l(0),
            t(0),
            r(-1),
            b(-1)
        {}

        long top (
        ) const { return t; }

        long& top (
        ) { return t; }

        void set_top (
            long top_
        ) { t = top_; }

        long left (
        ) const { return l; }

        long& left (
        ) { return l; }

        void set_left (
            long left_
        ) { l = left_; }

        long right (
        ) const { return r; }

        long& right (
        ) { return r; }

        void set_right (
            long right_
        ) { r = right_; }

        long bottom (
        ) const { return b; }

        long& bottom (
        ) { return b; }

        void set_bottom (
            long bottom_
        ) { b = bottom_; }
       
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

        unsigned long area (
        ) const
        {
            return width()*height();
        }

        bool is_empty (
        ) const { return (t > b || l > r); }

        rectangle operator + (
            const rectangle& rhs
        ) const
        {
            if (rhs.is_empty())
                return *this;
            else if (is_empty())
                return rhs;

            return rectangle (
                std::min(l,rhs.l),
                std::min(t,rhs.t),
                std::max(r,rhs.r),
                std::max(b,rhs.b)
                );
        }

        rectangle intersect (
            const rectangle& rhs
        ) const
        {
            return rectangle (
                std::max(l,rhs.l),
                std::max(t,rhs.t),
                std::min(r,rhs.r),
                std::min(b,rhs.b)
                );
        }

        bool contains (
            long x,
            long y
        ) const
        {
            if (x < l || x > r || y < t || y > b)
                return false;
            return true;
        }

        bool contains (
            const rectangle& rect
        ) const
        {
            return (rect + *this == *this);
        }

        rectangle& operator+= (
            const rectangle& rect
        )
        {
            *this = *this + rect;
            return *this;
        }

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
        long l;
        long t;
        long r;
        long b;   
    };

    inline std::ostream& operator<< (
        std::ostream& out, 
        const rectangle& item 
    )   
    {
        out << "[(" << item.left() << ", " << item.top() << ") (" << item.right() << ", " << item.bottom() << ")]";
        return out;
    }

// ----------------------------------------------------------------------------------------

    inline rectangle intersect (
        const rectangle& a,
        const rectangle& b
    ) { return a.intersect(b); }

// ----------------------------------------------------------------------------------------

    inline unsigned long area (
        const rectangle& a
    ) { return a.area(); }

// ----------------------------------------------------------------------------------------

    inline const rectangle shrink_rect (
        const rectangle& rect,
        long num 
    )
    {
        return rectangle(rect.left()+num, rect.top()+num, rect.right()-num, rect.bottom()-num);
    }

// ----------------------------------------------------------------------------------------

    inline const rectangle grow_rect (
        const rectangle& rect,
        long num 
    )
    {
        return shrink_rect(rect, -num);
    }

// ----------------------------------------------------------------------------------------

    inline const rectangle shrink_rect (
        const rectangle& rect,
        long width,
        long height
    )
    {
        return rectangle(rect.left()+width, rect.top()+height, rect.right()-width, rect.bottom()-height);
    }

// ----------------------------------------------------------------------------------------

    inline const rectangle grow_rect (
        const rectangle& rect,
        long width,
        long height
    )
    {
        return shrink_rect(rect, -width, -height);
    }

// ----------------------------------------------------------------------------------------

    inline const rectangle translate_rect (
        const rectangle& rect,
        long x,
        long y
    )
    {
        rectangle result;
        result.set_top ( rect.top() + y );
        result.set_bottom ( rect.bottom() + y );
        result.set_left ( rect.left() + x );
        result.set_right ( rect.right() + x );
        return result;
    }

// ----------------------------------------------------------------------------------------

    inline const rectangle resize_rect (
        const rectangle& rect,
        unsigned long width,
        unsigned long height
    )
    {
        return rectangle(rect.left(),rect.top(), 
                         rect.left()+width-1,
                         rect.top()+height-1);
    }

// ----------------------------------------------------------------------------------------

    inline const rectangle resize_rect_width (
        const rectangle& rect,
        unsigned long width
    )
    {
        return rectangle(rect.left(),rect.top(), 
                         rect.left()+width-1,
                         rect.bottom());
    }

// ----------------------------------------------------------------------------------------

    inline const rectangle resize_rect_height (
        const rectangle& rect,
        unsigned long height 
    )
    {
        return rectangle(rect.left(),rect.top(), 
                         rect.right(),
                         rect.top()+height-1);
    }

// ----------------------------------------------------------------------------------------

    inline const rectangle move_rect (
        const rectangle& rect,
        long x,
        long y 
    )
    {
        return rectangle(x, y, x+rect.width()-1, y+rect.height()-1);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T 
        >
    inline const rectangle get_rect (
        const T& m
    )
    {
        return rectangle(0, 0, num_columns(m)-1, num_rows(m)-1);
    }


// ----------------------------------------------------------------------------------------

}

#endif // DLIB_RECTANGLe_


