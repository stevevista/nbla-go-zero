// Copyright (C) 2003  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_ALGs_
#define DLIB_ALGs_

// this file contains miscellaneous stuff                      

// Give people who forget the -std=c++11 option a reminder
#if (defined(__GNUC__) && ((__GNUC__ >= 4 && __GNUC_MINOR__ >= 8) || (__GNUC__ > 4))) || \
    (defined(__clang__) && ((__clang_major__ >= 3 && __clang_minor__ >= 4) || (__clang_major__ >= 3)))
    #if __cplusplus < 201103
        #error "Dlib requires C++11 support.  Give your compiler the -std=c++11 option to enable it."
    #endif
#endif

#if defined __NVCC__
    // Disable the "statement is unreachable" message since it will go off on code that is
    // actually reachable but just happens to not be reachable sometimes during certain
    // template instantiations.
    #pragma diag_suppress code_is_unreachable
#endif


#ifdef _MSC_VER

#if  _MSC_VER < 1900
#error "dlib versions newer than v19.1 use C++11 and therefore require Visual Studio 2015 or newer."
#endif

// Disable the following warnings for Visual Studio

// this is to disable the "'this' : used in base member initializer list"
// warning you get from some of the GUI objects since all the objects
// require that their parent class be passed into their constructor. 
// In this case though it is totally safe so it is ok to disable this warning.
#pragma warning(disable : 4355)

// This is a warning you get sometimes when Visual Studio performs a Koenig Lookup. 
// This is a bug in visual studio.  It is a totally legitimate thing to 
// expect from a compiler. 
#pragma warning(disable : 4675)

// This is a warning you get from visual studio 2005 about things in the standard C++
// library being "deprecated."  I checked the C++ standard and it doesn't say jack 
// about any of them (I checked the searchable PDF).   So this warning is total Bunk.
#pragma warning(disable : 4996)

// This is a warning you get from visual studio 2003:
//    warning C4345: behavior change: an object of POD type constructed with an initializer 
//    of the form () will be default-initialized.
// I love it when this compiler gives warnings about bugs in previous versions of itself. 
#pragma warning(disable : 4345)


// Disable warnings about conversion from size_t to unsigned long and long.
#pragma warning(disable : 4267)

// Disable warnings about conversion from double to float  
#pragma warning(disable : 4244)
#pragma warning(disable : 4305)

// Disable "warning C4180: qualifier applied to function type has no meaning; ignored".
// This warning happens often in generic code that works with functions and isn't useful.
#pragma warning(disable : 4180)

// Disable "warning C4290: C++ exception specification ignored except to indicate a function is not __declspec(nothrow)"
#pragma warning(disable : 4290)


// DNN module uses template-based network declaration that leads to very long
// type names. Visual Studio will produce Warning C4503 in such cases. https://msdn.microsoft.com/en-us/library/074af4b6.aspx says
// that correct binaries are still produced even when this warning happens, but linker errors from visual studio, if they occurr could be confusing.
#pragma warning( disable: 4503 )


#endif

#ifdef __BORLANDC__
// Disable the following warnings for the Borland Compilers
//
// These warnings just say that the compiler is refusing to inline functions with
// loops or try blocks in them.  
//
#pragma option -w-8027
#pragma option -w-8026 
#endif

#include <string>       // for the exceptions
#include <algorithm>    // for std::swap
#include <new>          // for std::bad_alloc
#include <cstdlib>
#include <limits> // for std::numeric_limits for is_finite()
#include "../assert.h"
#include "../error.h"
#include "../enable_if.h"

// ----------------------------------------------------------------------------------------



namespace dlib
{

    template <
        typename T
        >
    class memory_manager_stateless_kernel_1
    {
        /*!      
            this implementation just calls new and delete directly
        !*/
        
        public:

            typedef T type;
            const static bool is_stateless = true;

            template <typename U>
            struct rebind {
                typedef memory_manager_stateless_kernel_1<U> other;
            };

            memory_manager_stateless_kernel_1(
            )
            {}

            virtual ~memory_manager_stateless_kernel_1(
            ) {}

            T* allocate (
            )
            {
                return new T; 
            }

            void deallocate (
                T* item
            )
            {
                delete item;
            }

            T* allocate_array (
                unsigned long size
            ) 
            { 
                return new T[size];
            }

            void deallocate_array (
                T* item
            ) 
            { 
                delete [] item;
            }

            void swap (memory_manager_stateless_kernel_1&)
            {}

        private:

            // restricted functions
            memory_manager_stateless_kernel_1(memory_manager_stateless_kernel_1&);        // copy constructor
            memory_manager_stateless_kernel_1& operator=(memory_manager_stateless_kernel_1&);    // assignment operator
    };

    template <
        typename T
        >
    inline void swap (
        memory_manager_stateless_kernel_1<T>& a, 
        memory_manager_stateless_kernel_1<T>& b 
    ) { a.swap(b); }   

// ----------------------------------------------------------------------------------------

    /*!A default_memory_manager

        This memory manager just calls new and delete directly.  

    !*/
    typedef memory_manager_stateless_kernel_1<char> default_memory_manager;

// ----------------------------------------------------------------------------------------

    /*!A swap !*/
    // make swap available in the dlib namespace
    using std::swap;

    /*!
        uint64 is a typedef for an unsigned integer that is exactly 64 bits wide.
        uint32 is a typedef for an unsigned integer that is exactly 32 bits wide.
        uint16 is a typedef for an unsigned integer that is exactly 16 bits wide.
        uint8  is a typedef for an unsigned integer that is exactly 8  bits wide.

        int64 is a typedef for an integer that is exactly 64 bits wide.
        int32 is a typedef for an integer that is exactly 32 bits wide.
        int16 is a typedef for an integer that is exactly 16 bits wide.
        int8  is a typedef for an integer that is exactly 8  bits wide.
    !*/


#ifdef __GNUC__
    typedef unsigned long long uint64;
    typedef long long int64;
#elif defined(__BORLANDC__)
    typedef unsigned __int64 uint64;
    typedef __int64 int64;
#elif defined(_MSC_VER)
    typedef unsigned __int64 uint64;
    typedef __int64 int64;
#else
    typedef unsigned long long uint64;
    typedef long long int64;
#endif

    typedef unsigned short uint16;
    typedef unsigned int   uint32;
    typedef unsigned char  uint8;

    typedef short int16;
    typedef int   int32;
    typedef char  int8;


    // make sure these types have the right sizes on this platform
    COMPILE_TIME_ASSERT(sizeof(uint8)  == 1);
    COMPILE_TIME_ASSERT(sizeof(uint16) == 2);
    COMPILE_TIME_ASSERT(sizeof(uint32) == 4);
    COMPILE_TIME_ASSERT(sizeof(uint64) == 8);

    COMPILE_TIME_ASSERT(sizeof(int8)  == 1);
    COMPILE_TIME_ASSERT(sizeof(int16) == 2);
    COMPILE_TIME_ASSERT(sizeof(int32) == 4);
    COMPILE_TIME_ASSERT(sizeof(int64) == 8);



    template <typename T, size_t s = sizeof(T)>
    struct unsigned_type;
    template <typename T>
    struct unsigned_type<T,1> { typedef uint8 type; };
    template <typename T>
    struct unsigned_type<T,2> { typedef uint16 type; };
    template <typename T>
    struct unsigned_type<T,4> { typedef uint32 type; };
    template <typename T>
    struct unsigned_type<T,8> { typedef uint64 type; };


// ----------------------------------------------------------------------------------------

    class noncopyable
    {
        /*!
            This class makes it easier to declare a class as non-copyable.
            If you want to make an object that can't be copied just inherit
            from this object.
        !*/

    protected:
        noncopyable() = default;
        ~noncopyable() = default;
    private:  // emphasize the following members are private
        noncopyable(const noncopyable&);
        const noncopyable& operator=(const noncopyable&);

    };

// ----------------------------------------------------------------------------------------

    template <
        typename T
        >
    void exchange (
        T& a,
        T& b
    )
    /*!
        This function does the exact same thing that global swap does and it does it by
        just calling swap.  But a lot of compilers have problems doing a Koenig Lookup
        and the fact that this has a different name (global swap has the same name as
        the member functions called swap) makes them compile right.

        So this is a workaround but not too ugly of one.  But hopefully I get get
        rid of this in a few years.  So this function is already deprecated. 

        This also means you should NOT use this function in your own code unless
        you have to support an old buggy compiler that benefits from this hack.
    !*/
    {
        using std::swap;
        using dlib::swap;
        swap(a,b);
    }

// ----------------------------------------------------------------------------------------

    /*!A is_pointer_type

        This is a template where is_pointer_type<T>::value == true when T is a pointer 
        type and false otherwise.
    !*/

    template <
        typename T
        >
    class is_pointer_type
    {
    public:
        enum { value = false };
    private:
        is_pointer_type();
    };

    template <
        typename T
        >
    class is_pointer_type<T*>
    {
    public:
        enum { value = true };
    private:
        is_pointer_type();
    };

// ----------------------------------------------------------------------------------------

    /*!A is_const_type

        This is a template where is_const_type<T>::value == true when T is a const 
        type and false otherwise.
    !*/

    template <typename T>
    struct is_const_type
    {
        static const bool value = false;
    };
    template <typename T>
    struct is_const_type<const T>
    {
        static const bool value = true;
    };
    template <typename T>
    struct is_const_type<const T&>
    {
        static const bool value = true;
    };

// ----------------------------------------------------------------------------------------

    /*!A is_reference_type 

        This is a template where is_reference_type<T>::value == true when T is a reference 
        type and false otherwise.
    !*/

    template <typename T>
    struct is_reference_type
    {
        static const bool value = false;
    };

    template <typename T> struct is_reference_type<const T&> { static const bool value = true; };
    template <typename T> struct is_reference_type<T&> { static const bool value = true; };

// ----------------------------------------------------------------------------------------

    /*!A is_same_type 

        This is a template where is_same_type<T,U>::value == true when T and U are the
        same type and false otherwise.   
    !*/

    template <
        typename T,
        typename U
        >
    class is_same_type
    {
    public:
        enum {value = false};
    private:
        is_same_type();
    };

    template <typename T>
    class is_same_type<T,T>
    {
    public:
        enum {value = true};
    private:
        is_same_type();
    };

// ----------------------------------------------------------------------------------------

    /*!A is_float_type

        This is a template that can be used to determine if a type is one of the built
        int floating point types (i.e. float, double, or long double).
    !*/

    template < typename T > struct is_float_type  { const static bool value = false; };
    template <> struct is_float_type<float>       { const static bool value = true; };
    template <> struct is_float_type<double>      { const static bool value = true; };
    template <> struct is_float_type<long double> { const static bool value = true; };

// ----------------------------------------------------------------------------------------

    /*!A is_convertible

        This is a template that can be used to determine if one type is convertible 
        into another type.

        For example:
            is_convertible<int,float>::value == true    // because ints are convertible to floats
            is_convertible<int*,float>::value == false  // because int pointers are NOT convertible to floats
    !*/

    template <typename from, typename to>
    struct is_convertible
    {
        struct yes_type { char a; };
        struct no_type { yes_type a[2]; };
        static const from& from_helper();
        static yes_type test(to);
        static no_type test(...);
        const static bool value = sizeof(test(from_helper())) == sizeof(yes_type);
    };

// ----------------------------------------------------------------------------------------

    struct general_ {};
    struct special_ : general_ {};
    template<typename> struct int_ { typedef int type; };

// ----------------------------------------------------------------------------------------


    /*!A is_same_object 

        This is a templated function which checks if both of its arguments are actually
        references to the same object.  It returns true if they are and false otherwise.

    !*/

    // handle the case where T and U are unrelated types.
    template < typename T, typename U >
    typename disable_if_c<is_convertible<T*, U*>::value || is_convertible<U*,T*>::value, bool>::type
    is_same_object (
        const T& a, 
        const U& b
    ) 
    { 
        return ((void*)&a == (void*)&b); 
    }

    // handle the case where T and U are related types because their pointers can be
    // implicitly converted into one or the other.  E.g. a derived class and its base class. 
    // Or where both T and U are just the same type.  This way we make sure that if there is a
    // valid way to convert between these two pointer types then we will take that route rather
    // than the void* approach used otherwise.
    template < typename T, typename U >
    typename enable_if_c<is_convertible<T*, U*>::value || is_convertible<U*,T*>::value, bool>::type
    is_same_object (
        const T& a, 
        const U& b
    ) 
    { 
        return (&a == &b); 
    }

// ----------------------------------------------------------------------------------------

    /*!A is_unsigned_type 

        This is a template where is_unsigned_type<T>::value == true when T is an unsigned
        scalar type and false when T is a signed scalar type.
    !*/
    template <
        typename T
        >
    struct is_unsigned_type
    {
        static const bool value = static_cast<T>((static_cast<T>(0)-static_cast<T>(1))) > 0;
    };
    template <> struct is_unsigned_type<long double> { static const bool value = false; };
    template <> struct is_unsigned_type<double>      { static const bool value = false; };
    template <> struct is_unsigned_type<float>       { static const bool value = false; };

// ----------------------------------------------------------------------------------------

    /*!A is_signed_type 

        This is a template where is_signed_type<T>::value == true when T is a signed
        scalar type and false when T is an unsigned scalar type.
    !*/
    template <
        typename T
        >
    struct is_signed_type
    {
        static const bool value = !is_unsigned_type<T>::value;
    };

// ----------------------------------------------------------------------------------------

    /*!A static_switch

        To use this template you give it some number of boolean expressions and it
        tells you which one of them is true.   If more than one of them is true then
        it causes a compile time error.

        for example:
            static_switch<1 + 1 == 2, 4 - 1 == 4>::value == 1  // because the first expression is true
            static_switch<1 + 1 == 3, 4 == 4>::value == 2      // because the second expression is true
            static_switch<1 + 1 == 3, 4 == 5>::value == 0      // 0 here because none of them are true 
            static_switch<1 + 1 == 2, 4 == 4>::value == compiler error  // because more than one expression is true 
    !*/

    template < bool v1 = 0, bool v2 = 0, bool v3 = 0, bool v4 = 0, bool v5 = 0,
               bool v6 = 0, bool v7 = 0, bool v8 = 0, bool v9 = 0, bool v10 = 0, 
               bool v11 = 0, bool v12 = 0, bool v13 = 0, bool v14 = 0, bool v15 = 0 >
    struct static_switch; 

    template <> struct static_switch<0,0,0,0,0,0,0,0,0,0,0,0,0,0,0> { const static int value = 0; };
    template <> struct static_switch<1,0,0,0,0,0,0,0,0,0,0,0,0,0,0> { const static int value = 1; };
    template <> struct static_switch<0,1,0,0,0,0,0,0,0,0,0,0,0,0,0> { const static int value = 2; };
    template <> struct static_switch<0,0,1,0,0,0,0,0,0,0,0,0,0,0,0> { const static int value = 3; };
    template <> struct static_switch<0,0,0,1,0,0,0,0,0,0,0,0,0,0,0> { const static int value = 4; };
    template <> struct static_switch<0,0,0,0,1,0,0,0,0,0,0,0,0,0,0> { const static int value = 5; };
    template <> struct static_switch<0,0,0,0,0,1,0,0,0,0,0,0,0,0,0> { const static int value = 6; };
    template <> struct static_switch<0,0,0,0,0,0,1,0,0,0,0,0,0,0,0> { const static int value = 7; };
    template <> struct static_switch<0,0,0,0,0,0,0,1,0,0,0,0,0,0,0> { const static int value = 8; };
    template <> struct static_switch<0,0,0,0,0,0,0,0,1,0,0,0,0,0,0> { const static int value = 9; };
    template <> struct static_switch<0,0,0,0,0,0,0,0,0,1,0,0,0,0,0> { const static int value = 10; };
    template <> struct static_switch<0,0,0,0,0,0,0,0,0,0,1,0,0,0,0> { const static int value = 11; };
    template <> struct static_switch<0,0,0,0,0,0,0,0,0,0,0,1,0,0,0> { const static int value = 12; };
    template <> struct static_switch<0,0,0,0,0,0,0,0,0,0,0,0,1,0,0> { const static int value = 13; };
    template <> struct static_switch<0,0,0,0,0,0,0,0,0,0,0,0,0,1,0> { const static int value = 14; };
    template <> struct static_switch<0,0,0,0,0,0,0,0,0,0,0,0,0,0,1> { const static int value = 15; };

// ----------------------------------------------------------------------------------------
    /*!A is_built_in_scalar_type
        
        This is a template that allows you to determine if the given type is a built
        in scalar type such as an int, char, float, short, etc.

        For example, is_built_in_scalar_type<char>::value == true
        For example, is_built_in_scalar_type<std::string>::value == false 
    !*/

    template <typename T> struct is_built_in_scalar_type        { const static bool value = false; };

    template <> struct is_built_in_scalar_type<float>           { const static bool value = true; };
    template <> struct is_built_in_scalar_type<double>          { const static bool value = true; };
    template <> struct is_built_in_scalar_type<long double>     { const static bool value = true; };
    template <> struct is_built_in_scalar_type<short>           { const static bool value = true; };
    template <> struct is_built_in_scalar_type<int>             { const static bool value = true; };
    template <> struct is_built_in_scalar_type<long>            { const static bool value = true; };
    template <> struct is_built_in_scalar_type<unsigned short>  { const static bool value = true; };
    template <> struct is_built_in_scalar_type<unsigned int>    { const static bool value = true; };
    template <> struct is_built_in_scalar_type<unsigned long>   { const static bool value = true; };
    template <> struct is_built_in_scalar_type<uint64>          { const static bool value = true; };
    template <> struct is_built_in_scalar_type<int64>           { const static bool value = true; };
    template <> struct is_built_in_scalar_type<char>            { const static bool value = true; };
    template <> struct is_built_in_scalar_type<signed char>     { const static bool value = true; };
    template <> struct is_built_in_scalar_type<unsigned char>   { const static bool value = true; };
    // Don't define one for wchar_t when using a version of visual studio
    // older than 8.0 (visual studio 2005) since before then they improperly set
    // wchar_t to be a typedef rather than its own type as required by the C++ 
    // standard.
#if !defined(_MSC_VER) || _NATIVE_WCHAR_T_DEFINED
    template <> struct is_built_in_scalar_type<wchar_t>         { const static bool value = true; };
#endif

// ----------------------------------------------------------------------------------------
    
    template <
        typename T
        >
    typename enable_if<is_built_in_scalar_type<T>,bool>::type is_finite (
        const T& value
    )
    /*!
        requires
            - value must be some kind of scalar type such as int or double
        ensures
            - returns true if value is a finite value (e.g. not infinity or NaN) and false
              otherwise.
    !*/
    {
        if (is_float_type<T>::value)
            return -std::numeric_limits<T>::infinity() < value && value < std::numeric_limits<T>::infinity();
        else
            return true;
    }

// ----------------------------------------------------------------------------------------

    /*!A basic_type

        This is a template that takes a type and strips off any const, volatile, or reference
        qualifiers and gives you back the basic underlying type.  So for example:

        basic_type<const int&>::type == int
    !*/

    template <typename T> struct basic_type { typedef T type; };
    template <typename T> struct basic_type<const T> { typedef T type; };
    template <typename T> struct basic_type<const T&> { typedef T type; };
    template <typename T> struct basic_type<volatile const T&> { typedef T type; };
    template <typename T> struct basic_type<T&> { typedef T type; };
    template <typename T> struct basic_type<volatile T&> { typedef T type; };
    template <typename T> struct basic_type<volatile T> { typedef T type; };
    template <typename T> struct basic_type<volatile const T> { typedef T type; };

// ----------------------------------------------------------------------------------------

    /*!A tabs 

        This is a template to compute the absolute value a number at compile time.

        For example,
            abs<-4>::value == 4
            abs<4>::value == 4
    !*/

        template <long x, typename enabled=void>
        struct tabs { const static long value = x; };
        template <long x>
        struct tabs<x,typename enable_if_c<(x < 0)>::type> { const static long value = -x; };

// ----------------------------------------------------------------------------------------

    /*!A tmax

        This is a template to compute the max of two values at compile time

        For example,
            abs<4,7>::value == 7
    !*/

        template <long x, long y, typename enabled=void>
        struct tmax { const static long value = x; };
        template <long x, long y>
        struct tmax<x,y,typename enable_if_c<(y > x)>::type> { const static long value = y; };

// ----------------------------------------------------------------------------------------

    /*!A tmin 

        This is a template to compute the min of two values at compile time

        For example,
            abs<4,7>::value == 4
    !*/

        template <long x, long y, typename enabled=void>
        struct tmin { const static long value = x; };
        template <long x, long y>
        struct tmin<x,y,typename enable_if_c<(y < x)>::type> { const static long value = y; };


}

#endif // DLIB_ALGs_

