#pragma once

#include "variable.hpp"
#include <iostream>
#include <functional>
#include <string>

namespace nblapp {

using std::vector;
using std::pair;
using std::string;
using std::shared_ptr;

struct ScopeData;
typedef shared_ptr<ScopeData> ScopeDataPtr;

class ParameterScope {
    ScopeDataPtr prev_scope;
public:
    ParameterScope(const string& name);
    ~ParameterScope();

    static Variable get_parameter(const std::string& key);
    static void set_parameter(const std::string& key, Variable param);
    static Variable get_parameter_or_create(const std::string& name, const vector<int>& shape, bool need_grad, std::function<Variable(const vector<int>&)> creator);
    static Variable get_or_create(const std::string& name, const vector<int>& shape, bool need_grad=true);
    static Variable get_or_create_constant(const std::string& name, const vector<int>& shape, float fill, bool need_grad=true);
    static Variable get_or_create_uniform(const std::string& name, const vector<int>& shape, float low=0, float high=1, bool need_grad=true);
    static Variable get_or_create_normal(const std::string& name, const vector<int>& shape, float mu=0, float sigma=1, bool need_grad=true);

    static void get_parameters(vector<pair<string, Variable>>& params, const string& path, bool grad_only);
    static vector<pair<string, Variable>> get_parameters(const string& path="", bool grad_only=true);
    static void clear_parameters();
    static void save_parameters(std::ostream& os);
    static bool load_parameters(std::istream& is);
    static bool save_parameters(const string& scope, const string& path);
    static bool load_parameters(const string& scope, const string& path);
};



}


