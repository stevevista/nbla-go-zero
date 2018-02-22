/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef UTILS_H_DEFINED
#define UTILS_H_DEFINED

#include "config.h"

#include <atomic>
#include <limits>
#include <string>

#include "ThreadPool.h"

extern Utils::ThreadPool thread_pool;

namespace Utils {
    void myprintf(const char *fmt, ...);

    template<class T>
    void atomic_add(std::atomic<T> &f, T d) {
        T old = f.load();
        while (!f.compare_exchange_weak(old, old + d));
    }

    template<typename T>
    T rotl(const T x, const int k) {
	    return (x << k) | (x >> (std::numeric_limits<T>::digits - k));
    }

    inline bool is7bit(int c) {
        return c >= 0 && c <= 127;
    }

    template <typename T, typename... Args>
    std::string format(const std::string &format, T first, Args... rest) {
      int size = snprintf(nullptr, 0, format.c_str(), first, rest...);
      std::vector<char> buffer(size + 1);
      snprintf(buffer.data(), size + 1, format.c_str(), first, rest...);
      return std::string(buffer.data(), buffer.data() + size);
    }
    
    /** String formatter without format.
    */
    inline std::string format(const std::string &format) {
      for (auto itr = format.begin(); itr != format.end(); itr++) {
        if (*itr == '%') {
          if (*(itr + 1) == '%') {
            itr++;
          } else {
          }
        }
      }
      return format;
    }

    namespace algorithm {

        inline bool find_first(const std::string& Input, const std::string& Search) {
            return Input.find(Search) != std::string::npos;
        }

        inline bool starts_with(const std::string& Input, const std::string& Search) {
            return Input.find(Search) == 0;
        }
    }
}

#endif
