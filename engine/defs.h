
#pragma once

// For windows support
#if defined(_MSC_VER) && !defined(__CUDACC__)
#ifdef goengine_EXPORTS
#define NENG_API __declspec(dllexport)
#else
#define NENG_API __declspec(dllimport)
#endif
#else
#define NENG_API
#endif
