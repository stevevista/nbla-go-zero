#pragma once

#include <string>

namespace boost {
    namespace algorithm {

        inline bool find_first(const std::string& Input, const std::string& Search) {
            return Input.find(Search) != std::string::npos;
        }

        inline bool starts_with(const std::string& Input, const std::string& Search) {
            return Input.find(Search) == 0;
        }
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
}
