// Copyright (C) 2005  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_RECTANGLe_
#define DLIB_RECTANGLe_

#include "../algs.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include "vector.h"
#include "generic_image.h"

namespace dlib
{

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
            const point& p
        ) :
            l(p.x()),
            t(p.y()),
            r(p.x()),
            b(p.y())
        {
        }

        rectangle (
            const point& p1,
            const point& p2
        )
        {
            *this = rectangle(p1) + rectangle(p2);
        }

        template <typename T>
        rectangle (
            const vector<T,2>& p1,
            const vector<T,2>& p2
        )
        {
            *this = rectangle(p1) + rectangle(p2);
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

        const point tl_corner (
        ) const { return point(left(), top()); }

        const point bl_corner (
        ) const { return point(left(), bottom()); } 

        const point tr_corner (
        ) const { return point(right(), top()); }

        const point br_corner (
        ) const { return point(right(), bottom()); }
       
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
            const point& p
        ) const
        {
            if (p.x() < l || p.x() > r || p.y() < t || p.y() > b)
                return false;
            return true;
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
            const point& p 
        )
        {
            *this = *this + rectangle(p);
            return *this;
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

    inline point center (
        const dlib::rectangle& rect
    )
    {
        point temp(rect.left() + rect.right() + 1,
                   rect.top() + rect.bottom() + 1);

        if (temp.x() < 0)
            temp.x() -= 1;

        if (temp.y() < 0)
            temp.y() -= 1;

        return temp/2;
    }

// ----------------------------------------------------------------------------------------

    inline dlib::vector<double,2> dcenter (
        const dlib::rectangle& rect
    )
    {
        dlib::vector<double,2> temp(rect.left() + rect.right(),
                                    rect.top() + rect.bottom());

        return temp/2.0;
    }

// ----------------------------------------------------------------------------------------

    inline const point nearest_point (
        const rectangle& rect,
        const point& p
    )
    {
        point temp(p);
        if (temp.x() < rect.left())
            temp.x() = rect.left();
        else if (temp.x() > rect.right())
            temp.x() = rect.right();

        if (temp.y() < rect.top())
            temp.y() = rect.top();
        else if (temp.y() > rect.bottom())
            temp.y() = rect.bottom();

        return temp;
    }

// ----------------------------------------------------------------------------------------

    inline size_t nearest_rect (
        const std::vector<rectangle>& rects,
        const point& p
    )
    {
        DLIB_ASSERT(rects.size() > 0);
        size_t idx = 0;
        double best_dist = std::numeric_limits<double>::infinity();

        for (size_t i = 0; i < rects.size(); ++i)
        {
            if (rects[i].contains(p))
            {
                return i;
            }
            else
            {
                double dist = (nearest_point(rects[i],p)-p).length();
                if (dist < best_dist)
                {
                    best_dist = dist;
                    idx = i;
                }
            }
        }
        return idx;
    }

// ----------------------------------------------------------------------------------------

    inline void clip_line_to_rectangle (
        const rectangle& box,
        point& p1,
        point& p2
    )
    {
        // Now clip the line segment so it is contained inside box.
        if (p1.x() == p2.x())
        {
            if (!box.contains(p1))
                p1.y() = box.top();
            if (!box.contains(p2))
                p2.y() = box.bottom();
        }
        else if (p1.y() == p2.y())
        {
            if (!box.contains(p1))
                p1.x() = box.left();
            if (!box.contains(p2))
                p2.x() = box.right();
        }
        else
        {
            // We use these relations to find alpha values.  These values tell us
            // how to generate points intersecting the rectangle boundaries.  We then
            // test the resulting points for ones that are inside the rectangle and output
            // those.
            //box.left()  == alpha1*(p1.x()-p2.x()) + p2.x();
            //box.right() == alpha2*(p1.x()-p2.x()) + p2.x();

            const point d = p1-p2;
            double alpha1 = (box.left()  -p2.x())/(double)d.x();
            double alpha2 = (box.right() -p2.x())/(double)d.x();
            double alpha3 = (box.top()   -p2.y())/(double)d.y();
            double alpha4 = (box.bottom()-p2.y())/(double)d.y();

            const point c1 = alpha1*d + p2;
            const point c2 = alpha2*d + p2;
            const point c3 = alpha3*d + p2;
            const point c4 = alpha4*d + p2;

            if (!box.contains(p1))
                p1 = c1;
            if (!box.contains(p2))
                p2 = c2;
            if (box.contains(c3))
            {
                if (!box.contains(p2))
                    p2 = c3;
                else if (!box.contains(p1))
                    p1 = c3;
            }
            if (box.contains(c4))
            {
                if (!box.contains(p2))
                    p2 = c4;
                else if (!box.contains(p1))
                    p1 = c4;
            }
        }

        p1 = nearest_point(box, p1);
        p2 = nearest_point(box, p2);
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

    inline rectangle operator+ (
        const rectangle& r,
        const point& p
    )
    {
        return r + rectangle(p);
    }

// ----------------------------------------------------------------------------------------

    inline rectangle operator+ (
        const point& p,
        const rectangle& r
    )
    {
        return r + rectangle(p);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_RECTANGLe_


