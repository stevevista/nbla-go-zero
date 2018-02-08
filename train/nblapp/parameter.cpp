
#include "parameter.hpp"
#include <sstream>
#include <cassert>
#include <unordered_map>
#include <nbla/exception.hpp>
#include <fstream>

namespace nblapp {

using std::make_shared;
using nbla::format_string;
using nbla::Exception;


vector<string> split_str(const string& name, char delim, bool strip) {

    std::stringstream ss(name);
    std::string item;
    std::vector<std::string> names;
    while (std::getline(ss, item, delim)) {
        names.push_back(item);
    }

    if (strip) {
        if (names.size() && names.front().empty())
            names.erase(names.begin());

        if (names.size() && names.back().empty())
            names.erase(names.end()-1); 
    }

    return names;
}


string join_str(vector<string>::const_iterator begin, vector<string>::const_iterator end, char delim) {
    std::string s;
    for (auto it=begin; it!=end; it++) {
        s += *it;
        s += delim;
    }
    if (!s.empty())
        s.erase(s.end()-1);
    return s;
}


struct ScopeData {
    std::unordered_map<std::string, Variable> params;
    std::unordered_map<std::string, ScopeDataPtr> children;
};

static ScopeDataPtr current_scope = make_shared<ScopeData>();
static ScopeDataPtr root_scope = current_scope;


ParameterScope::ParameterScope(const std::string& name) {

    auto names = split_str(name, '/', true);
    NBLA_CHECK(!names.empty(), nbla::error_code::unclassified, "Invalid argument of parameter_scope('%s').", name.c_str());

    prev_scope = current_scope;
    auto scope = current_scope;

    for (const auto& name : names) {
        auto it = scope->children.find(name);
        if (it != scope->children.end()) {
            scope = it->second;
        } else {
            auto s = make_shared<ScopeData>();
            scope->children.insert({name, s});
            scope = s;
        }
    }

    current_scope = scope;

}


ParameterScope::~ParameterScope() {
    current_scope = prev_scope;
}


Variable ParameterScope::get_parameter(const string& key) {

    auto names = split_str(key, '/', true);
    if (names.size()>1) {
        ParameterScope _(names[0]);
        return get_parameter(join_str(names.begin()+1, names.end(), '/'));
    }

    auto it = current_scope->params.find(key);
    if (it != current_scope->params.end()) {
        return it->second;
    }

    return Variable();
}


void ParameterScope::set_parameter(const string& key, Variable param) {

    auto names = split_str(key, '/', true);
    if (names.size()>1) {
        ParameterScope _(names[0]);
        set_parameter(join_str(names.begin()+1, names.end(), '/'), param);
        return;
    }
    
    current_scope->params.insert({key, param});
}


Variable ParameterScope::get_parameter_or_create(const std::string& name, const vector<int>& shape, bool need_grad, std::function<Variable(const vector<int>&)> creator) {

    auto names = split_str(name, '/', true);
    if (names.size()>1) {
        ParameterScope _(names[0]);
        return get_parameter_or_create(join_str(names.begin()+1, names.end(), '/'), shape, need_grad, creator);
    }

    auto param = get_parameter(name);
    if (!param) {
        param = creator(shape);
        assert(param.shape() == shape);
        param.set_need_grad(need_grad);
        set_parameter(name, param);
    } else {
        assert(param.shape() == shape);
        if (need_grad != param.need_grad()) {
            param = param.unlinked();
            param.set_need_grad(need_grad);
        }
    }

    return param;
}


Variable ParameterScope::get_or_create(const std::string& name, const vector<int>& shape, bool need_grad) {
    
    return get_parameter_or_create(name, shape, need_grad, [&](const vector<int>& shape) {
        return Variable(shape, need_grad);
    });
}

Variable ParameterScope::get_or_create_constant(const string& name, const vector<int>& shape, float fill, bool need_grad) {

    return get_parameter_or_create(name, shape, need_grad, [&](const vector<int>& shape) {
        return Variable::constant(shape, fill);
    });
}

Variable ParameterScope::get_or_create_uniform(const std::string& name, const vector<int>& shape, float low, float high, bool need_grad) {

    return get_parameter_or_create(name, shape, need_grad, [&](const vector<int>& shape) {
        return Variable::uniform(shape, low, high);
    });
}

Variable ParameterScope::get_or_create_normal(const std::string& name, const vector<int>& shape, float mu, float sigma, bool need_grad) {

    return get_parameter_or_create(name, shape, need_grad, [&](const vector<int>& shape) {
        return Variable::normal(shape, mu, sigma);
    });
}


void ParameterScope::get_parameters(vector<pair<string, Variable>>& params, const string& path, bool grad_only) {
    
        for (const auto& kv : current_scope->children) {
            const auto& k = kv.first;
            ParameterScope _(k);
            get_parameters(params, path.empty() ? k : (path + "/" + k), grad_only);
        }
    
        for (const auto& kv : current_scope->params) {
            const auto& k = kv.first;
            auto& v = kv.second;
            if (!grad_only || v.need_grad()) {
                params.push_back({path.empty() ? k : (path + "/" + k), v});
            }
        }
}
    
    
vector<pair<string, Variable>> ParameterScope::get_parameters(const string& path, bool grad_only) {
        vector<pair<string, Variable>> out;
        get_parameters(out, path, grad_only);
        return out;
}
    
    
void ParameterScope::clear_parameters() {
        current_scope->params.clear();
        current_scope->children.clear();
}



static void emit_bool(std::ostream& os, bool b);
static void emit_string(std::ostream& os, const string& str);
static void emit_variable(std::ostream& os, const Variable& var);
static bool restore_bool(std::istream& is, bool& b);
static bool restore_string(std::istream& is, string& str);
static bool restore_variable_dim(std::istream& is, vector<int>& shape);
static bool restore_variable(std::istream& is, Variable& var);


bool ParameterScope::save_parameters(const string& scope, const string& path) {
    std::ofstream ofs(path, std::ofstream::binary);
    if (!ofs)
        return false;

    ParameterScope _(scope);
    save_parameters(ofs);
    return true;
}


bool ParameterScope::load_parameters(const string& scope, const string& path) {
    std::ifstream ifs(path, std::ofstream::binary);
    if (!ifs)
        return false;

    ParameterScope _(scope);
    return load_parameters(ifs);
}

void ParameterScope::save_parameters(std::ostream& os) {

    auto params = get_parameters("", false);
    for (const auto& kv : params) {
        emit_string(os, kv.first);
        emit_bool(os, kv.second.need_grad());
        emit_variable(os, kv.second);
    }
}


bool ParameterScope::load_parameters(std::istream& is) {

    int count = 0;
    while(true) {
        string key;
        vector<int> shape;
        bool need_grad;

        if (!restore_string(is, key))
            break;

        if (!restore_bool(is, need_grad))
            return false;

        if (!restore_variable_dim(is, shape))
            return false;

        auto var = get_or_create(key, shape, need_grad);
        if (!restore_variable(is, var))
            return false;

        count++;
    }

    return count > 0;
}


//
//
//

static void emit_bool(std::ostream& os, bool b) {
  char c = b ? 1 : 0;
  os.write(&c, 1);
}

static void emit_string(std::ostream& os, const string& str) {
  int sz = str.size();
  os.write((char*)&sz, 4);
  os.write(str.c_str(), sz);
}

static void emit_variable(std::ostream& os, const Variable& var) {

  const auto shape = var.shape();
  const int dim = shape.size();
  os.write((char*)&dim, 4);
  int sz = 1;
  for (int d : shape) {
    os.write((char*)&d, 4);
    sz *= d;
  }
  auto data = var.data<float>();
  os.write((char*)data, sz*sizeof(float));
}


static bool restore_bool(std::istream& is, bool& b) {

    char c;
    if (is.read(&c, 1).gcount() != 1)
        return false;

    b = c;
    return true;
}

static bool restore_string(std::istream& is, string& str) {

  int sz;
  is.read((char*)&sz, 4);
  if (sz <=0 || sz > 1024)
    return false;

  str.resize(sz);
  if (is.read(&str[0], sz).gcount() != sz)
    return false;

  return true;
}

static bool restore_variable_dim(std::istream& is, vector<int>& shape) {

  int dim;
  if (is.read((char*)&dim, 4).gcount() != 4)
    return false;

  for (int i=0; i<dim; i++) {
    int d;
    if (is.read((char*)&d, 4).gcount() != 4)
      return false;
    shape.push_back(d);
  }

  return true;
}


static bool restore_variable(std::istream& is, Variable& var) {

  int sz = var.size();
  auto dst = var.data<float>();
  if (is.read((char*)dst, sz*sizeof(float)).gcount() != sz*sizeof(float))
    return false;

  return true;
}



    

}
