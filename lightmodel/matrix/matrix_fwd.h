// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_FWD
#define DLIB_MATRIx_FWD

#include "algs.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct row_major_layout;

// ----------------------------------------------------------------------------------------

    template <
        typename T,
        long num_rows = 0,
        long num_cols = 0,
        typename mem_manager = default_memory_manager,
        typename layout = row_major_layout 
        >
    class matrix; 


    template <typename T, typename helper = void>
    struct is_matrix
    {
        /*!
            - if (T is some kind of matrix expression from the matrix/matrix_exp_abstract.h component) then
                - is_matrix<T>::value == true
            - else
                - is_matrix<T>::value == false
        !*/

        static const bool value = false;
        // Don't set the helper to anything.  Just let it be void.
        ASSERT_ARE_SAME_TYPE(helper,void);
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_FWD

