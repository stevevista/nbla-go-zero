#pragma once 

// For windows support
#if defined(_MSC_VER) && !defined(__CUDACC__)
#ifdef nblapp_EXPORTS
#define NPP_API __declspec(dllexport)
#else
#define NPP_API __declspec(dllimport)
#endif
#else
#define NPP_API
#endif
