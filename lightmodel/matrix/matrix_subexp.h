// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_SUBEXP_
#define DLIB_MATRIx_SUBEXP_


#include "matrix_op.h"
#include "matrix.h"
#include "matrix_expressions.h"
#include "matrix_mat.h"



namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
    
    template <long start, long inc, long end>
    const matrix_range_static_exp<start,inc,end> range (
    ) 
    { 
        COMPILE_TIME_ASSERT(inc > 0);
        return matrix_range_static_exp<start,inc,end>(); 
    }

    template <long start, long end>
    const matrix_range_static_exp<start,1,end> range (
    ) 
    { 
        return matrix_range_static_exp<start,1,end>(); 
    }

    inline const matrix_range_exp<long> range (
        long start,
        long end
    ) 
    { 
        return matrix_range_exp<long>(start,end); 
    }

    inline const matrix_range_exp<long> range (
        long start,
        long inc,
        long end
    ) 
    { 
        DLIB_ASSERT(inc > 0, 
            "\tconst matrix_exp range(start, inc, end)"
            << "\n\tInvalid inputs to this function"
            << "\n\tstart: " << start 
            << "\n\tinc:   " << inc
            << "\n\tend:   " << end
            );

        return matrix_range_exp<long>(start,inc,end); 
    }

// ----------------------------------------------------------------------------------------

    template <typename M>
    struct op_rowm 
    {
        op_rowm(const M& m_, const long& row_) : m(m_), row(row_) {}
        const M& m;
        const long row;

        const static long cost = M::cost;
        const static long NR = 1;
        const static long NC = M::NC;
        typedef typename M::type type;
        typedef typename M::const_ret_type const_ret_type;
        typedef typename M::mem_manager_type mem_manager_type;
        typedef typename M::layout_type layout_type;
        const_ret_type apply ( long, long c) const { return m(row,c); }

        long nr () const { return 1; }
        long nc () const { return m.nc(); }

        template <typename U> bool aliases               ( const matrix_exp<U>& item) const { return m.aliases(item); }
        template <typename U> bool destructively_aliases ( const matrix_exp<U>& item) const { return m.aliases(item); }
    };

    template <
        typename EXP
        >
    const matrix_op<op_rowm<EXP> > rowm (
        const matrix_exp<EXP>& m,
        long row
    )
    {
        DLIB_ASSERT(row >= 0 && row < m.nr(), 
            "\tconst matrix_exp rowm(const matrix_exp& m, row)"
            << "\n\tYou have specified invalid sub matrix dimensions"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\trow:    " << row 
            );

        typedef op_rowm<EXP> op;
        return matrix_op<op>(op(m.ref(),row));
    }

    template <typename EXP>
    struct rowm_exp
    {
        typedef matrix_op<op_rowm<EXP> > type;
    };

// ----------------------------------------------------------------------------------------

    template <typename M>
    struct op_rowm2 
    {
        op_rowm2(const M& m_, const long& row_, const long& len) : m(m_), row(row_), length(len) {}
        const M& m;
        const long row;
        const long length;

        const static long cost = M::cost;
        const static long NR = 1;
        const static long NC = 0;
        typedef typename M::type type;
        typedef typename M::const_ret_type const_ret_type;
        typedef typename M::mem_manager_type mem_manager_type;
        typedef typename M::layout_type layout_type;
        const_ret_type apply ( long , long c) const { return m(row,c); }

        long nr () const { return 1; }
        long nc () const { return length; }

        template <typename U> bool aliases               ( const matrix_exp<U>& item) const { return m.aliases(item); }
        template <typename U> bool destructively_aliases ( const matrix_exp<U>& item) const { return m.aliases(item); }
    };

    template <
        typename EXP
        >
    const matrix_op<op_rowm2<EXP> > rowm (
        const matrix_exp<EXP>& m,
        long row,
        long length
    )
    {
        DLIB_ASSERT(row >= 0 && row < m.nr() && 
                    length >= 0 && length <= m.nc(), 
            "\tconst matrix_exp rowm(const matrix_exp& m, row, length)"
            << "\n\tYou have specified invalid sub matrix dimensions"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\trow:    " << row 
            << "\n\tlength: " << length 
            );

        typedef op_rowm2<EXP> op;
        return matrix_op<op>(op(m.ref(), row, length));
    }

// ----------------------------------------------------------------------------------------

    template <typename M1, typename M2>
    struct op_rowm_range 
    {
        op_rowm_range( const M1& m1_, const M2& rows_) : m1(m1_), rows(rows_) {}
        const M1& m1;
        const M2& rows;

        const static long cost = M1::cost+M2::cost;
        typedef typename M1::type type;
        typedef typename M1::const_ret_type const_ret_type;
        typedef typename M1::mem_manager_type mem_manager_type;
        typedef typename M1::layout_type layout_type;
        const static long NR = M2::NC*M2::NR;
        const static long NC = M1::NC;

        const_ret_type apply ( long r, long c) const { return m1(rows(r),c); }

        long nr () const { return rows.size(); }
        long nc () const { return m1.nc(); }

        template <typename U> bool aliases               ( const matrix_exp<U>& item) const 
        { return m1.aliases(item) || rows.aliases(item); }
        template <typename U> bool destructively_aliases ( const matrix_exp<U>& item) const 
        { return m1.aliases(item) || rows.aliases(item); }
    };

    template <
        typename EXP1,
        typename EXP2
        >
    const matrix_op<op_rowm_range<EXP1,EXP2> > rowm (
        const matrix_exp<EXP1>& m,
        const matrix_exp<EXP2>& rows
    )
    {
        // the rows matrix must contain integer elements 
        COMPILE_TIME_ASSERT(std::numeric_limits<typename EXP2::type>::is_integer);

        DLIB_ASSERT(0 <= min(rows) && max(rows) < m.nr() && (rows.nr() == 1 || rows.nc() == 1), 
            "\tconst matrix_exp rowm(const matrix_exp& m, const matrix_exp& rows)"
            << "\n\tYou have given invalid arguments to this function"
            << "\n\tm.nr():     " << m.nr()
            << "\n\tm.nc():     " << m.nc() 
            << "\n\tmin(rows):  " << min(rows) 
            << "\n\tmax(rows):  " << max(rows) 
            << "\n\trows.nr():  " << rows.nr()
            << "\n\trows.nc():  " << rows.nc()
            );

        typedef op_rowm_range<EXP1,EXP2> op;
        return matrix_op<op>(op(m.ref(),rows.ref()));
    }

// ----------------------------------------------------------------------------------------

    template <typename M>
    struct op_colm 
    {
        op_colm(const M& m_, const long& col_) : m(m_), col(col_) {}
        const M& m;
        const long col;

        const static long cost = M::cost;
        const static long NR = M::NR;
        const static long NC = 1;
        typedef typename M::type type;
        typedef typename M::const_ret_type const_ret_type;
        typedef typename M::mem_manager_type mem_manager_type;
        typedef typename M::layout_type layout_type;
        const_ret_type apply ( long r, long) const { return m(r,col); }

        long nr () const { return m.nr(); }
        long nc () const { return 1; }

        template <typename U> bool aliases               ( const matrix_exp<U>& item) const { return m.aliases(item); }
        template <typename U> bool destructively_aliases ( const matrix_exp<U>& item) const { return m.aliases(item); }
    };

    template <
        typename EXP
        >
    const matrix_op<op_colm<EXP> > colm (
        const matrix_exp<EXP>& m,
        long col 
    )
    {
        DLIB_ASSERT(col >= 0 && col < m.nc(), 
            "\tconst matrix_exp colm(const matrix_exp& m, row)"
            << "\n\tYou have specified invalid sub matrix dimensions"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\tcol:    " << col 
            );

        typedef op_colm<EXP> op;
        return matrix_op<op>(op(m.ref(),col));
    }

    template <typename EXP>
    struct colm_exp
    {
        typedef matrix_op<op_colm<EXP> > type;
    };

// ----------------------------------------------------------------------------------------

    template <typename M>
    struct op_colm2 
    {
        op_colm2(const M& m_, const long& col_, const long& len) : m(m_), col(col_), length(len) {}
        const M& m;
        const long col;
        const long length;

        const static long cost = M::cost;
        const static long NR = 0;
        const static long NC = 1;
        typedef typename M::type type;
        typedef typename M::const_ret_type const_ret_type;
        typedef typename M::mem_manager_type mem_manager_type;
        typedef typename M::layout_type layout_type;
        const_ret_type apply ( long r, long ) const { return m(r,col); }

        long nr () const { return length; }
        long nc () const { return 1; }

        template <typename U> bool aliases               ( const matrix_exp<U>& item) const { return m.aliases(item); }
        template <typename U> bool destructively_aliases ( const matrix_exp<U>& item) const { return m.aliases(item); }
    };

    template <
        typename EXP
        >
    const matrix_op<op_colm2<EXP> > colm (
        const matrix_exp<EXP>& m,
        long col,
        long length
    )
    {
        DLIB_ASSERT(col >= 0 && col < m.nc() && 
                    length >= 0 && length <= m.nr(), 
            "\tconst matrix_exp colm(const matrix_exp& m, col, length)"
            << "\n\tYou have specified invalid sub matrix dimensions"
            << "\n\tm.nr(): " << m.nr()
            << "\n\tm.nc(): " << m.nc() 
            << "\n\tcol:    " << col 
            << "\n\tlength: " << length 
            );

        typedef op_colm2<EXP> op;
        return matrix_op<op>(op(m.ref(),col, length));
    }

// ----------------------------------------------------------------------------------------

    template <typename M1, typename M2>
    struct op_colm_range 
    {
        op_colm_range( const M1& m1_, const M2& cols_) : m1(m1_), cols(cols_) {}
        const M1& m1;
        const M2& cols;

        typedef typename M1::type type;
        typedef typename M1::const_ret_type const_ret_type;
        typedef typename M1::mem_manager_type mem_manager_type;
        typedef typename M1::layout_type layout_type;
        const static long NR = M1::NR;
        const static long NC = M2::NC*M2::NR;
        const static long cost = M1::cost+M2::cost;

        const_ret_type apply (long r, long c) const { return m1(r,cols(c)); }

        long nr () const { return m1.nr(); }
        long nc () const { return cols.size(); }

        template <typename U> bool aliases               ( const matrix_exp<U>& item) const 
        { return m1.aliases(item) || cols.aliases(item); }
        template <typename U> bool destructively_aliases ( const matrix_exp<U>& item) const 
        { return m1.aliases(item) || cols.aliases(item); }
    };

    template <
        typename EXP1,
        typename EXP2
        >
    const matrix_op<op_colm_range<EXP1,EXP2> > colm (
        const matrix_exp<EXP1>& m,
        const matrix_exp<EXP2>& cols
    )
    {
        // the rows matrix must contain integer elements 
        COMPILE_TIME_ASSERT(std::numeric_limits<typename EXP2::type>::is_integer);

        DLIB_ASSERT(0 <= min(cols) && max(cols) < m.nc() && (cols.nr() == 1 || cols.nc() == 1), 
            "\tconst matrix_exp colm(const matrix_exp& m, const matrix_exp& cols)"
            << "\n\tYou have given invalid arguments to this function"
            << "\n\tm.nr():     " << m.nr()
            << "\n\tm.nc():     " << m.nc() 
            << "\n\tmin(cols):  " << min(cols) 
            << "\n\tmax(cols):  " << max(cols) 
            << "\n\tcols.nr():  " << cols.nr()
            << "\n\tcols.nc():  " << cols.nc()
            );

        typedef op_colm_range<EXP1,EXP2> op;
        return matrix_op<op>(op(m.ref(),cols.ref()));
    }

// ----------------------------------------------------------------------------------------

    template <typename T>
    class assignable_ptr_matrix
    {
    public:
        typedef T type;
        typedef row_major_layout layout_type;
        typedef matrix<T,0,0,default_memory_manager,layout_type> matrix_type;

        assignable_ptr_matrix(
            T* ptr_,
            long nr_,
            long nc_
        ) : ptr(ptr_), height(nr_), width(nc_){}

        T& operator() (
            long r,
            long c
        )
        {
            return ptr[r*width + c];
        }

        const T& operator() (
            long r,
            long c
        ) const
        {
            return ptr[r*width + c];
        }

        long nr() const { return height; }
        long nc() const { return width; }

        template <typename EXP>
        assignable_ptr_matrix& operator= (
            const matrix_exp<EXP>& exp
        ) 
        {
            // You can only assign to a set_ptrm() expression with a source matrix that
            // contains the same type of elements as the target (i.e. you can't mix double
            // and float types).
            COMPILE_TIME_ASSERT((is_same_type<T, typename EXP::type>::value == true));

            DLIB_ASSERT( exp.nr() == height && exp.nc() == width,
                "\tassignable_matrix_expression set_ptrm()"
                << "\n\tYou have tried to assign to this object using a matrix that isn't the right size"
                << "\n\texp.nr() (source matrix): " << exp.nr()
                << "\n\texp.nc() (source matrix): " << exp.nc() 
                << "\n\twidth (target matrix):    " << width
                << "\n\theight (target matrix):   " << height
                );

            if (exp.destructively_aliases(mat(ptr,height,width)) == false)
            {
                matrix_assign(*this, exp); 
            }
            else
            {
                // make a temporary copy of the matrix we are going to assign to ptr to 
                // avoid aliasing issues during the copy
                this->operator=(tmp(exp));
            }

            return *this;
        }

        template <typename EXP>
        assignable_ptr_matrix& operator+= (
            const matrix_exp<EXP>& exp
        ) 
        {
            // You can only assign to a set_ptrm() expression with a source matrix that
            // contains the same type of elements as the target (i.e. you can't mix double
            // and float types).
            COMPILE_TIME_ASSERT((is_same_type<T, typename EXP::type>::value == true));

            DLIB_ASSERT( exp.nr() == height && exp.nc() == width,
                "\tassignable_matrix_expression set_ptrm()"
                << "\n\tYou have tried to assign to this object using a matrix that isn't the right size"
                << "\n\texp.nr() (source matrix): " << exp.nr()
                << "\n\texp.nc() (source matrix): " << exp.nc() 
                << "\n\twidth (target matrix):    " << width
                << "\n\theight (target matrix):   " << height
                );

            if (exp.destructively_aliases(mat(ptr,height,width)) == false)
            {
                matrix_assign(*this, mat(ptr,height,width)+exp); 
            }
            else
            {
                // make a temporary copy of the matrix we are going to assign to ptr to 
                // avoid aliasing issues during the copy
                this->operator+=(tmp(exp));
            }

            return *this;
        }

        template <typename EXP>
        assignable_ptr_matrix& operator-= (
            const matrix_exp<EXP>& exp
        ) 
        {
            // You can only assign to a set_ptrm() expression with a source matrix that
            // contains the same type of elements as the target (i.e. you can't mix double
            // and float types).
            COMPILE_TIME_ASSERT((is_same_type<T, typename EXP::type>::value == true));

            DLIB_ASSERT( exp.nr() == height && exp.nc() == width,
                "\tassignable_matrix_expression set_ptrm()"
                << "\n\tYou have tried to assign to this object using a matrix that isn't the right size"
                << "\n\texp.nr() (source matrix): " << exp.nr()
                << "\n\texp.nc() (source matrix): " << exp.nc() 
                << "\n\twidth (target matrix):    " << width
                << "\n\theight (target matrix):   " << height
                );

            if (exp.destructively_aliases(mat(ptr,height,width)) == false)
            {
                matrix_assign(*this, mat(ptr,height,width)-exp); 
            }
            else
            {
                // make a temporary copy of the matrix we are going to assign to ptr to 
                // avoid aliasing issues during the copy
                this->operator-=(tmp(exp));
            }

            return *this;
        }

        assignable_ptr_matrix& operator= (
            const T& value
        )
        {
            const long size = width*height;
            for (long i = 0; i < size; ++i)
                ptr[i] = value;

            return *this;
        }

        assignable_ptr_matrix& operator+= (
            const T& value
        )
        {
            const long size = width*height;
            for (long i = 0; i < size; ++i)
                ptr[i] += value;

            return *this;
        }

        assignable_ptr_matrix& operator-= (
            const T& value
        )
        {
            const long size = width*height;
            for (long i = 0; i < size; ++i)
                ptr[i] -= value;

            return *this;
        }


        T* ptr;
        const long height;
        const long width;
    };


    template <typename T>
    assignable_ptr_matrix<T> set_ptrm (
        T* ptr,
        long nr,
        long nc = 1
    )
    {
        DLIB_ASSERT(nr >= 0 && nc >= 0, 
            "\t assignable_matrix_expression set_ptrm(T* ptr, long nr, long nc)"
            << "\n\t The dimensions can't be negative."
            << "\n\t nr: " << nr
            << "\n\t nc: " << nc
            );


        return assignable_ptr_matrix<T>(ptr,nr,nc);
    }

}

#endif // DLIB_MATRIx_SUBEXP_

