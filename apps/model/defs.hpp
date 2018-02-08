#pragma once 

// For windows support
#if defined(_MSC_VER) && !defined(__CUDACC__)
#ifdef model_EXPORTS
#define MODEL_API __declspec(dllexport)
#else
#define MODEL_API __declspec(dllimport)
#endif
#else
#define MODEL_API
#endif
