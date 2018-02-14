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

    struct num_con_outputs
    {
        num_con_outputs(unsigned long n) : num_outputs(n) {}
        unsigned long num_outputs;
    };

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
        static_assert(_nr >= 0, "The number of rows in a filter must be >= 0");
        static_assert(_nc >= 0, "The number of columns in a filter must be >= 0");
        static_assert(_stride_y > 0, "The filter stride must be > 0");
        static_assert(_stride_x > 0, "The filter stride must be > 0");
        static_assert(_nr==0 || (0 <= _padding_y && _padding_y < _nr), "The padding must be smaller than the filter size.");
        static_assert(_nc==0 || (0 <= _padding_x && _padding_x < _nc), "The padding must be smaller than the filter size.");
        static_assert(_nr!=0 || 0 == _padding_y, "If _nr==0 then the padding must be set to 0 as well.");
        static_assert(_nc!=0 || 0 == _padding_x, "If _nr==0 then the padding must be set to 0 as well.");

        con_(
            num_con_outputs o
        ) : 
            num_filters_(o.num_outputs),
            padding_y_(_padding_y),
            padding_x_(_padding_x)
        {
            DLIB_CASSERT(num_filters_ > 0);
        }

        con_() : con_(num_con_outputs(_num_filters)) {}

        con_ (
            const con_& item
        ) : 
            weights(item.weights),
            biases(item.biases),
            num_filters_(item.num_filters_),
            padding_y_(item.padding_y_),
            padding_x_(item.padding_x_)
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
            weights = item.weights;
            biases = item.biases;
            padding_y_ = item.padding_y_;
            padding_x_ = item.padding_x_;
            num_filters_ = item.num_filters_;
            return *this;
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

            weights.set_size(_num_filters, shape[1], _nr, _nc);
            std::copy(data.begin(), data.end(), weights.host_write_only());


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

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            conv.setup(sub.get_output(),
                        weights,
                       _stride_y,
                       _stride_x,
                       padding_y_,
                       padding_x_);
            conv(false, output,
                sub.get_output(), weights);

            if (bias_mode == FC_HAS_BIAS) {
                tt::add(1,output,1,biases);
            }
        } 


        friend std::ostream& operator<<(std::ostream& out, const con_& item)
        {
            out << "con\t ("
                << "num_filters="<<item.num_filters_
                << ", nr="<<item.nr()
                << ", nc="<<item.nc()
                << ", stride_y="<<_stride_y
                << ", stride_x="<<_stride_x
                << ", padding_y="<<item.padding_y_
                << ", padding_x="<<item.padding_x_
                << ")";
            return out;
        }

    private:

        resizable_tensor weights, biases;

        tt::tensor_conv conv;
        long num_filters_;

        // These are here only because older versions of con (which you might encounter
        // serialized to disk) used different padding settings.
        int padding_y_;
        int padding_x_;

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
        fc_(num_fc_outputs o) : num_outputs(o.num_outputs), num_inputs(0)
        {}

        fc_() : fc_(num_fc_outputs(num_outputs_)) {}

        unsigned long get_num_outputs (
        ) const { return num_outputs; }

        fc_bias_mode get_bias_mode (
        ) const { return bias_mode; }

        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {

            auto& shape = it->shape;
            auto& data = it->data;
            it++;

            // 
            if (shape.size() != 2 || 
                shape[1] != num_outputs_)
                throw std::runtime_error("Wrong weights shape found while deserializing dlib::fc_");

            num_inputs = shape[0];
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

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            DLIB_CASSERT((long)num_inputs == sub.get_output().nr()*sub.get_output().nc()*sub.get_output().k(),
                "The size of the input tensor to this fc layer doesn't match the size the fc layer was trained with.");
            output.set_size(sub.get_output().num_samples(), num_outputs);

            tt::gemm(0,output, 1,sub.get_output(),false, weights,false);
            if (bias_mode == FC_HAS_BIAS)
            {
                tt::add(1,output,1,biases);
            }
        } 

        friend std::ostream& operator<<(std::ostream& out, const fc_& item)
        {
            if (bias_mode == FC_HAS_BIAS)
            {
                out << "fc\t ("
                    << "num_outputs="<<item.num_outputs
                    << ")";
            }
            else
            {
                out << "fc_no_bias ("
                    << "num_outputs="<<item.num_outputs
                    << ")";
            }
            return out;
        }

    private:

        unsigned long num_outputs;
        unsigned long num_inputs;
        resizable_tensor weights, biases;
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
        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {
            
            resizable_tensor running_means, running_variances;
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

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::affine_transform_conv(output, input, gamma, beta);
        } 

        friend std::ostream& operator<<(std::ostream& out, const affine_& )
        {
            out << "affine";
            return out;
        }

    private:
        resizable_tensor gamma, beta;
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
        void setup (const SUBNET& /*sub*/)
        {
        }

        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {
            return it;
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            auto&& t1 = sub.get_output();
            auto&& t2 = layer<tag>(sub).get_output();
            output.set_size(std::max(t1.num_samples(),t2.num_samples()),
                            std::max(t1.k(),t2.k()),
                            std::max(t1.nr(),t2.nr()),
                            std::max(t1.nc(),t2.nc()));
            tt::add(output, t1, t2);
        }

        friend std::ostream& operator<<(std::ostream& out, const add_prev_& item)
        {
            out << "add_prev"<<id;
            return out;
        }

        friend void to_xml(const add_prev_& item, std::ostream& out)
        {
            out << "<add_prev tag='"<<id<<"'/>\n";
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
    template <typename SUBNET> using add_prev6  = add_prev<tag6, SUBNET>;
    template <typename SUBNET> using add_prev7  = add_prev<tag7, SUBNET>;
    template <typename SUBNET> using add_prev8  = add_prev<tag8, SUBNET>;
    template <typename SUBNET> using add_prev9  = add_prev<tag9, SUBNET>;
    template <typename SUBNET> using add_prev10 = add_prev<tag10, SUBNET>;

    using add_prev1_  = add_prev_<tag1>;
    using add_prev2_  = add_prev_<tag2>;
    using add_prev3_  = add_prev_<tag3>;
    using add_prev4_  = add_prev_<tag4>;
    using add_prev5_  = add_prev_<tag5>;
    using add_prev6_  = add_prev_<tag6>;
    using add_prev7_  = add_prev_<tag7>;
    using add_prev8_  = add_prev_<tag8>;
    using add_prev9_  = add_prev_<tag9>;
    using add_prev10_ = add_prev_<tag10>;

// ----------------------------------------------------------------------------------------

    class relu_
    {
    public:
        relu_() 
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {
            return it;
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::relu(output, input);
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

    class htan_
    {
    public:
        htan_() 
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
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

    // ----------------------------------------------------------------------------------------

    template <
        long _nr,
        long _nc,
        int _stride_y,
        int _stride_x,
        int _padding_y = 0,
        int _padding_x = 0
        >
    class avg_pool_
    {
    public:
        static_assert(_nr >= 0, "The number of rows in a filter must be >= 0");
        static_assert(_nc >= 0, "The number of columns in a filter must be >= 0");
        static_assert(_stride_y > 0, "The filter stride must be > 0");
        static_assert(_stride_x > 0, "The filter stride must be > 0");
        static_assert(0 <= _padding_y && ((_nr==0 && _padding_y == 0) || (_nr!=0 && _padding_y < _nr)), 
            "The padding must be smaller than the filter size, unless the filters size is 0.");
        static_assert(0 <= _padding_x && ((_nc==0 && _padding_x == 0) || (_nc!=0 && _padding_x < _nc)), 
            "The padding must be smaller than the filter size, unless the filters size is 0.");

        avg_pool_(
        ) :
            padding_y_(_padding_y),
            padding_x_(_padding_x)
        {}

        avg_pool_ (
            const avg_pool_& item
        )  :
            padding_y_(item.padding_y_),
            padding_x_(item.padding_x_)
        {
            // this->ap is non-copyable so we have to write our own copy to avoid trying to
            // copy it and getting an error.
        }

        avg_pool_& operator= (
            const avg_pool_& item
        )
        {
            if (this == &item)
                return *this;

            padding_y_ = item.padding_y_;
            padding_x_ = item.padding_x_;

            // this->ap is non-copyable so we have to write our own copy to avoid trying to
            // copy it and getting an error.
            return *this;
        }

        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {
            return it;
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            ap.setup_avg_pooling(_nr!=0?_nr:sub.get_output().nr(), 
                                _nc!=0?_nc:sub.get_output().nc(),
                                _stride_y, _stride_x, padding_y_, padding_x_);

            ap(output, sub.get_output());
        } 


        friend std::ostream& operator<<(std::ostream& out, const avg_pool_& item)
        {
            out << "avg_pool ("
                << "nr="<<_nr
                << ", nc="<<_nc
                << ", stride_y="<<_stride_y
                << ", stride_x="<<_stride_x
                << ", padding_y="<<item.padding_y_
                << ", padding_x="<<item.padding_x_
                << ")";
            return out;
        }

    private:

        tt::pooling ap;
        int padding_y_;
        int padding_x_;
    };

    template <
        long nr,
        long nc,
        int stride_y,
        int stride_x,
        int pading_y,
        int padding_x,
        typename SUBNET
        >
    using avg_pool = add_layer<avg_pool_<nr,nc,stride_y,stride_x,pading_y,padding_x>, SUBNET>;

}

#endif // DLIB_DNn_LAYERS_H_


