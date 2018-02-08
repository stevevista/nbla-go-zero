// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_LAYERS_H_
#define DLIB_DNn_LAYERS_H_

#include "tensor.h"
#include "core.h"
#include <iostream>
#include <string>
#include "tensor_tools.h"
#include <sstream>


namespace dlib
{
    enum fc_bias_mode
    {
        FC_HAS_BIAS = 0,
        FC_NO_BIAS = 1
    };

// ----------------------------------------------------------------------------------------
    template <
        long _num_filters,
        long _nr,
        long _nc,
        int _stride_y,
        int _stride_x,
        fc_bias_mode bias_mode = FC_NO_BIAS,
        int _padding_y = _nr/2,
        int _padding_x = _nc/2
        >
    class con_
    {
    public:
        static_assert(_num_filters > 0, "The number of filters must be > 0");
        static_assert(_nr > 0, "The number of rows in a filter must be > 0");
        static_assert(_nc > 0, "The number of columns in a filter must be > 0");
        static_assert(_stride_y > 0, "The filter stride must be > 0");
        static_assert(_stride_x > 0, "The filter stride must be > 0");

        con_() {}

        con_ (
            const con_& item
        ) : 
            params(item.params),
            biases(item.biases)
        {
            // this->conv is non-copyable and basically stateless, so we have to write our
            // own copy to avoid trying to copy it and getting an error.
        }

        con_& operator= (
            const con_& item
        )
        {
            if (this == &item)
                return *this;

            // this->conv is non-copyable and basically stateless, so we have to write our
            // own copy to avoid trying to copy it and getting an error.
            params = item.params;
            biases = item.biases;
            return *this;
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, tensor& output)
        {
            conv.setup(sub.get_output(),
                       params,
                       _stride_y,
                       _stride_x,
                       _padding_y,
                       _padding_x);
            conv(false, output,
                sub.get_output(), params);
            
            if (bias_mode == FC_HAS_BIAS) {
                tt::add(1,output,1,biases);
            }
        } 

        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {

            auto& shape = it->shape;
            auto& data = it->data;
            it++;

            // 
            if (shape.size() != 4 || 
                shape[0] != _num_filters ||
                shape[2] != _nr ||
                shape[3] != _nc)
                throw std::runtime_error("Wrong weights shape found while deserializing dlib::con_");

            params.set_size(_num_filters, shape[1], _nr, _nc);
            std::copy(data.begin(), data.end(), params.host_write_only());


            if (bias_mode == FC_HAS_BIAS) {

                auto& data = it->data;
                it++;

                if (data.size() != _num_filters)
                    throw std::runtime_error("Wrong weights shape found while deserializing dlib::con_bias");

                biases.set_size(1, _num_filters);
                std::copy(data.begin(), data.end(), biases.host_write_only());
            }

            return it;
        }

        friend std::ostream& operator<<(std::ostream& out, const con_& item)
        {
            out << "con\t ("
                << "num_filters="<<_num_filters
                << ", nr="<<_nr
                << ", nc="<<_nc
                << ", stride_y="<<_stride_y
                << ", stride_x="<<_stride_x
                << ")";
            return out;
        }

        
    private:
        tt::tensor_conv conv;
        tensor params;
        tensor biases;
    };

    template <
        long num_filters,
        long nr,
        long nc,
        int stride_y,
        int stride_x,
        typename SUBNET
        >
    using con = add_layer<con_<num_filters,nr,nc,stride_y,stride_x, FC_NO_BIAS>, SUBNET>;

    template <
        long num_filters,
        long nr,
        long nc,
        int stride_y,
        int stride_x,
        typename SUBNET
        >
    using con_bias = add_layer<con_<num_filters,nr,nc,stride_y,stride_x, FC_HAS_BIAS>, SUBNET>;

// ----------------------------------------------------------------------------------------

    struct num_fc_outputs
    {
        num_fc_outputs(unsigned long n) : num_outputs(n) {}
        unsigned long num_outputs;
    };

    template <
        unsigned long num_outputs_,
        fc_bias_mode bias_mode
        >
    class fc_
    {
        static_assert(num_outputs_ > 0, "The number of outputs from a fc_ layer must be > 0");

    public:

        fc_() {}

        template <typename SUBNET>
        void forward(const SUBNET& sub, tensor& output)
        {
            output.set_size(sub.get_output().num_samples(), num_outputs_);

            tt::gemm(0,output, 1,sub.get_output(),false, weights,false);
            if (bias_mode == FC_HAS_BIAS)
            {
                tt::add(1,output,1,biases);
            }
        } 

        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {

            auto& shape = it->shape;
            auto& data = it->data;
            it++;

            // 
            if (shape.size() != 2 || 
                shape[1] != num_outputs_)
                throw std::runtime_error("Wrong weights shape found while deserializing dlib::fc_");

            weights.set_size(shape[0], num_outputs_);
            std::copy(data.begin(), data.end(), weights.host_write_only());

            

            if (bias_mode == FC_HAS_BIAS) {

                auto& data = it->data;
                it++;

                if (data.size() != num_outputs_)
                    throw std::runtime_error("Wrong weights shape found while deserializing dlib::fc_");

                biases.set_size(1, num_outputs_);
                std::copy(data.begin(), data.end(), biases.host_write_only());
            }

            return it;
        }

        friend std::ostream& operator<<(std::ostream& out, const fc_& item)
        {
            if (bias_mode == FC_HAS_BIAS)
            {
                out << "fc\t ("
                    << "num_outputs="<< num_outputs_
                    << ")";
            }
            else
            {
                out << "fc_no_bias ("
                    << "num_outputs="<< num_outputs_
                    << ")";
            }
            return out;
        }

    private:
        tensor weights, biases;
    };

    template <
        unsigned long num_outputs,
        typename SUBNET
        >
    using fc = add_layer<fc_<num_outputs,FC_HAS_BIAS>, SUBNET>;

    template <
        unsigned long num_outputs,
        typename SUBNET
        >
    using fc_no_bias = add_layer<fc_<num_outputs,FC_NO_BIAS>, SUBNET>;

// ----------------------------------------------------------------------------------------

    class affine_
    {
    public:
        affine_(
        )
        {
        }


        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::affine_transform_conv(output, input, gamma, beta);
        } 

        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {
            
            tensor running_means, running_variances;
            const double eps=1e-05;

            for (int i=0; i<4; i++, it++) {
                auto& shape = it->shape;
                auto& data = it->data;

                if (shape.size() != 4 || 
                    shape[0] != 1 ||
                    shape[2] != 1 ||
                    shape[3] != 1)
                    throw std::runtime_error("Wrong weights shape found while deserializing dlib::affine_");

                
                if (i == 0) {
                    gamma.set_size(1, shape[1]);
                    std::copy(data.begin(), data.end(), gamma.host_write_only());
                } else if (i == 1) {
                    beta.set_size(1, shape[1]);
                    std::copy(data.begin(), data.end(), beta.host_write_only());
                } else if (i == 2) {
                    running_means.set_size(1, shape[1]);
                    std::copy(data.begin(), data.end(), running_means.host_write_only());
                } else if (i == 3) {
                    running_variances.set_size(1, shape[1]);
                    std::copy(data.begin(), data.end(), running_variances.host_write_only());
                }
            }

            gamma = pointwise_multiply(mat(gamma), 1.0f/sqrt(mat(running_variances)+eps));
            beta = mat(beta) - pointwise_multiply(mat(gamma), mat(running_means));

            return it;
        }

        friend std::ostream& operator<<(std::ostream& out, const affine_& )
        {
            out << "affine";
            return out;
        }

    private:
        tensor gamma, beta;
    };

    template <typename SUBNET>
    using affine = add_layer<affine_, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <
        template<typename> class tag
        >
    class add_prev_
    {
    public:
        const static unsigned long id = tag_id<tag>::id;

        add_prev_() 
        {
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, tensor& output)
        {
            auto&& t1 = sub.get_output();
            auto&& t2 = layer<tag>(sub).get_output();
            output.set_size(std::max(t1.num_samples(),t2.num_samples()),
                            std::max(t1.k(),t2.k()),
                            std::max(t1.nr(),t2.nr()),
                            std::max(t1.nc(),t2.nc()));
            tt::add(output, t1, t2);
        }

        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {
            return it;
        }

        friend std::ostream& operator<<(std::ostream& out, const add_prev_& item)
        {
            out << "add_prev"<<id;
            return out;
        }
    };

    template <
        template<typename> class tag,
        typename SUBNET
        >
    using add_prev = add_layer<add_prev_<tag>, SUBNET>;

    template <typename SUBNET> using add_prev1  = add_prev<tag1, SUBNET>;
    template <typename SUBNET> using add_prev2  = add_prev<tag2, SUBNET>;
    template <typename SUBNET> using add_prev3  = add_prev<tag3, SUBNET>;
    template <typename SUBNET> using add_prev4  = add_prev<tag4, SUBNET>;
    template <typename SUBNET> using add_prev5  = add_prev<tag5, SUBNET>;

    using add_prev1_  = add_prev_<tag1>;
    using add_prev2_  = add_prev_<tag2>;
    using add_prev3_  = add_prev_<tag3>;
    using add_prev4_  = add_prev_<tag4>;
    using add_prev5_  = add_prev_<tag5>;

// ----------------------------------------------------------------------------------------

    class relu_
    {
    public:
        relu_() 
        {
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::relu(output, input);
        } 


        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {
            return it;
        }

        friend std::ostream& operator<<(std::ostream& out, const relu_& )
        {
            out << "relu";
            return out;
        }
    };


    template <typename SUBNET>
    using relu = add_layer<relu_, SUBNET>;


// ----------------------------------------------------------------------------------------

    class sig_
    {
    public:
        sig_() 
        {
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::sigmoid(output, input);
        } 

        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {
            return it;
        }

        friend std::ostream& operator<<(std::ostream& out, const sig_& )
        {
            out << "sig";
            return out;
        }
    };


    template <typename SUBNET>
    using sig = add_layer<sig_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class htan_
    {
    public:
        htan_() 
        {
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::tanh(output, input);
        } 

        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {
            return it;
        }

        friend std::ostream& operator<<(std::ostream& out, const htan_& )
        {
            out << "htan";
            return out;
        }
    };


    template <typename SUBNET>
    using htan = add_layer<htan_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class softmax_
    {
    public:
        softmax_() : temprature(1)
        {
        }

        void set_temprature(double temp) {
            temprature = 1.0/temp;
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            if (temprature != 1)
                output *= temprature;
            tt::softmax(output, input);
        } 

        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {
            return it;
        }

        friend std::ostream& operator<<(std::ostream& out, const softmax_& )
        {
            out << "softmax";
            return out;
        }
    private:
        double temprature;
    };

    template <typename SUBNET>
    using softmax = add_layer<softmax_, SUBNET>;

}

#endif // DLIB_DNn_LAYERS_H_


