// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_MAT_Hh_
#define DLIB_MATRIx_MAT_Hh_

#include <vector>
#include "matrix_op.h"
#include "../array2d.h"
#include "generic_image.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------
    
    template <
        typename EXP
        >
    const matrix_exp<EXP>& mat (
        const matrix_exp<EXP>& m
    )
    {
        return m;
    }

// ----------------------------------------------------------------------------------------

    template <typename image_type, typename pixel_type>
    struct op_image_to_mat : does_not_alias 
    {
        op_image_to_mat( const image_type& img) : imgview(img){}

        const_image_view<image_type> imgview;

        const static long cost = 1;
        const static long NR = 0;
        const static long NC = 0;
        typedef pixel_type type;
        typedef const pixel_type& const_ret_type;
        typedef default_memory_manager mem_manager_type;
        typedef row_major_layout layout_type;

        const_ret_type apply (long r, long c ) const { return imgview[r][c]; }

        long nr () const { return imgview.nr(); }
        long nc () const { return imgview.nc(); }
    }; 

// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        > // The reason we disable this if it is a matrix is because this matrix_op claims
          // to not alias any matrix.  But obviously that would be a problem if we let it
          // take a matrix.
    const typename disable_if<is_matrix<image_type>,matrix_op<op_image_to_mat<image_type, typename image_traits<image_type>::pixel_type> > >::type mat (
        const image_type& img 
    )
    {
        typedef op_image_to_mat<image_type, typename image_traits<image_type>::pixel_type> op;
        return matrix_op<op>(op(img));
    }

// ----------------------------------------------------------------------------------------

    template <typename image_type>
    struct op_image_view_to_mat : does_not_alias 
    {
        op_image_view_to_mat( const image_view<image_type>& img) : imgview(img){}

        typedef typename image_traits<image_type>::pixel_type pixel_type;

        const image_view<image_type>& imgview;

        const static long cost = 1;
        const static long NR = 0;
        const static long NC = 0;
        typedef pixel_type type;
        typedef const pixel_type& const_ret_type;
        typedef default_memory_manager mem_manager_type;
        typedef row_major_layout layout_type;

        const_ret_type apply (long r, long c ) const { return imgview[r][c]; }

        long nr () const { return imgview.nr(); }
        long nc () const { return imgview.nc(); }
    }; 

    template <
        typename image_type
        > 
    const matrix_op<op_image_view_to_mat<image_type> > mat (
        const image_view<image_type>& img 
    )
    {
        typedef op_image_view_to_mat<image_type> op;
        return matrix_op<op>(op(img));
    }

// ----------------------------------------------------------------------------------------

    template <typename image_type>
    struct op_const_image_view_to_mat : does_not_alias 
    {
        op_const_image_view_to_mat( const const_image_view<image_type>& img) : imgview(img){}

        typedef typename image_traits<image_type>::pixel_type pixel_type;

        const const_image_view<image_type>& imgview;

        const static long cost = 1;
        const static long NR = 0;
        const static long NC = 0;
        typedef pixel_type type;
        typedef const pixel_type& const_ret_type;
        typedef default_memory_manager mem_manager_type;
        typedef row_major_layout layout_type;

        const_ret_type apply (long r, long c ) const { return imgview[r][c]; }

        long nr () const { return imgview.nr(); }
        long nc () const { return imgview.nc(); }
    }; 

    template <
        typename image_type
        > 
    const matrix_op<op_const_image_view_to_mat<image_type> > mat (
        const const_image_view<image_type>& img 
    )
    {
        typedef op_const_image_view_to_mat<image_type> op;
        return matrix_op<op>(op(img));
    }

// ----------------------------------------------------------------------------------------

    template <typename T>
    struct op_pointer_to_mat;   

    template <typename T>
    struct op_pointer_to_col_vect   
    {
        op_pointer_to_col_vect(
            const T* ptr_,
            const long size_
        ) : ptr(ptr_), size(size_){}

        const T* ptr;
        const long size;

        const static long cost = 1;
        const static long NR = 0;
        const static long NC = 1;
        typedef T type;
        typedef const T& const_ret_type;
        typedef default_memory_manager mem_manager_type;
        typedef row_major_layout layout_type;

        const_ret_type apply (long r, long ) const { return ptr[r]; }

        long nr () const { return size; }
        long nc () const { return 1; }

        template <typename U> bool aliases               ( const matrix_exp<U>& ) const { return false; }
        template <typename U> bool destructively_aliases ( const matrix_exp<U>& ) const { return false; }

        template <long num_rows, long num_cols, typename mem_manager, typename layout>
        bool aliases (
            const matrix_exp<matrix<T,num_rows,num_cols, mem_manager,layout> >& item
        ) const 
        { 
            if (item.size() == 0)
                return false;
            else
                return (ptr == &item(0,0)); 
        }

        inline bool aliases (
            const matrix_exp<matrix_op<op_pointer_to_mat<T> > >& item
        ) const;

        bool aliases (
            const matrix_exp<matrix_op<op_pointer_to_col_vect<T> > >& item
        ) const
        {
            return item.ref().op.ptr == ptr;
        }
    }; 

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    const matrix_op<op_pointer_to_col_vect<T> > mat (
        const T* ptr,
        long nr
    )
    {
        DLIB_ASSERT(nr >= 0 , 
                    "\tconst matrix_exp mat(ptr, nr)"
                    << "\n\t nr must be >= 0"
                    << "\n\t nr: " << nr
        );
        typedef op_pointer_to_col_vect<T> op;
        return matrix_op<op>(op(ptr, nr));
    }

// ----------------------------------------------------------------------------------------

    template <typename T>
    struct op_pointer_to_mat  
    {
        op_pointer_to_mat(
            const T* ptr_,
            const long nr_,
            const long nc_ 
        ) : ptr(ptr_), rows(nr_), cols(nc_), stride(nc_){}

        op_pointer_to_mat(
            const T* ptr_,
            const long nr_,
            const long nc_,
            const long stride_
        ) : ptr(ptr_), rows(nr_), cols(nc_), stride(stride_){}

        const T* ptr;
        const long rows;
        const long cols;
        const long stride;

        const static long cost = 1;
        const static long NR = 0;
        const static long NC = 0;
        typedef T type;
        typedef const T& const_ret_type;
        typedef default_memory_manager mem_manager_type;
        typedef row_major_layout layout_type;

        const_ret_type apply (long r, long c) const { return ptr[r*stride + c]; }

        long nr () const { return rows; }
        long nc () const { return cols; }

        template <typename U> bool aliases               ( const matrix_exp<U>& ) const { return false; }
        template <typename U> bool destructively_aliases ( const matrix_exp<U>& ) const { return false; }

        template <long num_rows, long num_cols, typename mem_manager, typename layout>
        bool aliases (
            const matrix_exp<matrix<T,num_rows,num_cols, mem_manager,layout> >& item
        ) const 
        { 
            if (item.size() == 0)
                return false;
            else
                return (ptr == &item(0,0)); 
        }

        bool aliases (
            const matrix_exp<matrix_op<op_pointer_to_mat<T> > >& item
        ) const
        {
            return item.ref().op.ptr == ptr;
        }

        bool aliases (
            const matrix_exp<matrix_op<op_pointer_to_col_vect<T> > >& item
        ) const
        {
            return item.ref().op.ptr == ptr;
        }
    }; 

    template <typename T>
    bool op_pointer_to_col_vect<T>::
    aliases (
        const matrix_exp<matrix_op<op_pointer_to_mat<T> > >& item
    ) const
    {
        return item.ref().op.ptr == ptr;
    }

    template <typename T, long NR, long NC, typename MM, typename L>
    bool matrix<T,NR,NC,MM,L>::aliases (
        const matrix_exp<matrix_op<op_pointer_to_mat<T> > >& item
    ) const
    {
        if (size() != 0)
            return item.ref().op.ptr == &data(0,0);
        else
            return false;
    }

    template <typename T, long NR, long NC, typename MM, typename L>
    bool matrix<T,NR,NC,MM,L>::aliases (
        const matrix_exp<matrix_op<op_pointer_to_col_vect<T> > >& item
    ) const
    {
        if (size() != 0)
            return item.ref().op.ptr == &data(0,0);
        else
            return false;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    const matrix_op<op_pointer_to_mat<T> > mat (
        const T* ptr,
        long nr,
        long nc
    )
    {
        DLIB_ASSERT(nr >= 0 && nc >= 0 , 
                    "\tconst matrix_exp mat(ptr, nr, nc)"
                    << "\n\t nr and nc must be >= 0"
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
        );
        typedef op_pointer_to_mat<T> op;
        return matrix_op<op>(op(ptr,nr,nc));
    }

    template <
        typename T
        >
    const matrix_op<op_pointer_to_mat<T> > mat (
        const T* ptr,
        long nr,
        long nc,
        long stride
    )
    {
        DLIB_ASSERT(nr >= 0 && nc >= 0 && stride > 0 , 
                    "\tconst matrix_exp mat(ptr, nr, nc, stride)"
                    << "\n\t nr and nc must be >= 0 while stride > 0"
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
                    << "\n\t stride: " << stride
        );
        typedef op_pointer_to_mat<T> op;
        return matrix_op<op>(op(ptr,nr,nc,stride));
    }

// ----------------------------------------------------------------------------------------

}


namespace dlib
{



// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    const matrix_op<op_pointer_to_mat<T> > pointer_to_matrix (
        const T* ptr,
        long nr,
        long nc
    )
    {
        DLIB_ASSERT(nr > 0 && nc > 0 , 
                    "\tconst matrix_exp pointer_to_matrix(ptr, nr, nc)"
                    << "\n\t nr and nc must be bigger than 0"
                    << "\n\t nr: " << nr
                    << "\n\t nc: " << nc
        );
        typedef op_pointer_to_mat<T> op;
        return matrix_op<op>(op(ptr,nr,nc));
    }

    template <
        typename T
        >
    const matrix_op<op_pointer_to_col_vect<T> > pointer_to_column_vector (
        const T* ptr,
        long nr
    )
    {
        DLIB_ASSERT(nr > 0 , 
                    "\tconst matrix_exp pointer_to_column_vector(ptr, nr)"
                    << "\n\t nr must be bigger than 0"
                    << "\n\t nr: " << nr
        );
        typedef op_pointer_to_col_vect<T> op;
        return matrix_op<op>(op(ptr, nr));
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    inline matrix<double,1,1> mat (
        double value
    )
    {
        matrix<double,1,1> temp;
        temp(0) = value;
        return temp;
    }

    inline matrix<float,1,1> mat (
        float value
    )
    {
        matrix<float,1,1> temp;
        temp(0) = value;
        return temp;
    }

    inline matrix<long double,1,1> mat (
        long double value
    )
    {
        matrix<long double,1,1> temp;
        temp(0) = value;
        return temp;
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_MAT_Hh_


