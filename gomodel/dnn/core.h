// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_CORE_H_
#define DLIB_DNn_CORE_H_

#include "tensor.h"
#include <iterator>
#include <memory>
#include <sstream>
#include <type_traits>
#include <utility>
#include <tuple>
#include <cmath>
#include <vector>
#include "tensor_tools.h"
#include <type_traits>

#ifdef _MSC_VER
// Tell Visual Studio not to recursively inline functions very much because otherwise it
// takes hours to compile the DNN code sometimes.  It's crazy.  Hopefully we can remove
// this some day when the visual studio compiler is more efficient.
#pragma inline_depth(2)
#endif


namespace dlib
{

    struct param_data {
        std::vector<int> shape;
        std::vector<float> data;
    };

    template <typename T>
    class input
    {
    public:
        typedef int input_type;
    };

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        // The reason we return an int for this version rather than doing the more straight forward thing (like we do above) is to avoid a bug in visual studio 2015.
        template <typename T>
        auto call_clean_method_if_exists (
            T& obj,
            special_
        ) -> typename int_<decltype(&T::clean)>::type { obj.clean();  return 0;  }

        template <typename T>
        void call_clean_method_if_exists (T& , general_) {}
    }
    template <typename T>
    void call_clean_method_if_exists(T& obj) { impl::call_clean_method_if_exists(obj, special_()); }


// ----------------------------------------------------------------------------------------

    namespace impl
    {
        class repeat_input_layer 
        {
            /*!
                None of the declarations in this object are really used. The only reason it
                exists is to allow the repeat object to use a special input layer in its
                internal networks which will cause add_tag_layer objects that happen to be
                right at the input to not create copies of their input tensors.  So
                introducing the repeat_input_layer object allows us to optimize the
                implementation of add_tag_layer for a special case that arises when it's
                used in the context of the repeat layer.
            !*/
        public:
            typedef int input_type;

            friend std::ostream& operator<<(std::ostream& out, const repeat_input_layer&) { return out; }
        };

        inline std::string tensor_to_str (
            const tensor& t,
            int& min_length 
        ) 
        {
            if (t.size() == 0)
                return "";

            std::ostringstream sout;
            sout << "output size=(num:"<<  t.num_samples() << ", ";
            sout << "k:" << t.k() << ",";
            while (sout.tellp() < 28) sout << " ";
            sout << "nr:" << t.nr() << ",";
            while (sout.tellp() < 28+8) sout << " ";
            sout << "nc:" << t.nc() << ")";
            while (sout.tellp() < min_length) sout << " ";
            min_length = sout.tellp();
            sout << "\t";
            return sout.str();
        }
    }

// ----------------------------------------------------------------------------------------

    // Tell us if T is one of the special layer types (i.e. add_layer, repeat, add_tag_layer, or
    // add_skip_layer).
    template <typename T> struct is_nonloss_layer_type : std::false_type {};
    // Tell us if T is an instance of add_loss_layer.
    template <typename T> struct is_loss_layer_type : std::false_type {};
    // Tell us if T is an instance of add_layer
    template <typename T> struct is_add_layer : std::false_type {};

    namespace impl
    {
        template <typename T> struct alwaysbool { typedef bool type; };
        // one more structure for VS 2015 UP3 support workaround
        template <typename T> struct alwaysbool2 { typedef bool type; };

        resizable_tensor& rt();

        template <typename layer_type, typename SUBNET>
        constexpr auto is_inplace_layer(
            layer_type& layer,
            const SUBNET& sub 
        ) -> typename alwaysbool2<decltype(layer.forward(sub,rt()))>::type
        {
            return false;
        }

        template <typename layer_type, typename SUBNET>
        constexpr auto is_inplace_layer(
            layer_type& layer,
            const SUBNET& sub
        ) -> typename alwaysbool<decltype(layer.forward_inplace(sub.get_output(),rt()))>::type
        {
            return true;
        }

        template <typename layer_type, typename SUBNET>
        auto call_layer_forward(
            layer_type& layer,
            const SUBNET& sub, 
            tensor& /*data_output*/
        ) -> decltype(layer.forward(sub,rt()))
        {
            // This overload of call_layer_forward() is here because this template
            // naturally gets instantiated but only on code paths that never get executed.
            // So rather than writing a bunch of hard to read template magic around call
            // sites we just have this overload that doesn't do anything (and an assert to
            // make sure that's the case).
            DLIB_CASSERT(false, "This should never happen");
        }

        template <typename layer_type, typename SUBNET>
        auto call_layer_forward(
            layer_type& layer,
            const SUBNET& sub, 
            resizable_tensor& data_output
        ) -> decltype(layer.forward(sub,data_output))
        {
            layer.forward(sub,data_output);
        }

        template <typename layer_type, typename SUBNET>
        auto call_layer_forward(
            layer_type& layer,
            const SUBNET& sub, 
            tensor& data_output
        ) -> decltype(layer.forward_inplace(sub.get_output(),data_output))
        {
            layer.forward_inplace(sub.get_output(),data_output);
        }

    } // end namespace impl


// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    namespace dimpl
    {
        template <typename T, bool is_first = true, typename enabled=void>
        class subnet_wrapper
        {
        public:
            subnet_wrapper(const subnet_wrapper&) = delete;
            subnet_wrapper& operator=(const subnet_wrapper&) = delete;

            subnet_wrapper(T& l_) : l(l_) {}
            // Not much here because in this case T is one of the input layer types 
            // that doesn't have anything in it.
            typedef T layer_details_type;
            const layer_details_type& layer_details() const { return l; }
        private:
            T& l;
        };

        template <typename T>
        class subnet_wrapper<T,true, typename std::enable_if<is_nonloss_layer_type<T>::value>::type>
        {

        public:
            subnet_wrapper(const subnet_wrapper&) = delete;
            subnet_wrapper& operator=(const subnet_wrapper&) = delete;

            typedef T wrapped_type;
            const static size_t num_computational_layers = T::num_computational_layers;
            const static size_t num_layers = T::num_layers;
            typedef typename T::layer_details_type layer_details_type;

            subnet_wrapper(T& l_) : l(l_),subnetwork(l.subnet()) {}

            const tensor& get_output() const { return l.private_get_output(); }

            const layer_details_type& layer_details() const { return l.layer_details(); }

            const subnet_wrapper<typename T::subnet_type,false>& subnet() const { return subnetwork; }
            subnet_wrapper<typename T::subnet_type,false>& subnet() { return subnetwork; }

        private:
            T& l;
            subnet_wrapper<typename T::subnet_type,false> subnetwork;
        };

        template <typename T>
        class subnet_wrapper<T,false, typename std::enable_if<is_nonloss_layer_type<T>::value>::type>
        {

        public:
            subnet_wrapper(const subnet_wrapper&) = delete;
            subnet_wrapper& operator=(const subnet_wrapper&) = delete;

            typedef T wrapped_type;
            const static size_t num_computational_layers = T::num_computational_layers;
            const static size_t num_layers = T::num_layers;
            typedef typename T::layer_details_type layer_details_type;

            subnet_wrapper(T& l_) : l(l_),subnetwork(l.subnet()) {}

            const tensor& get_output() const { return l.get_output(); }

            const layer_details_type& layer_details() const { return l.layer_details(); }

            const subnet_wrapper<typename T::subnet_type,false>& subnet() const { return subnetwork; }
            subnet_wrapper<typename T::subnet_type,false>& subnet() { return subnetwork; }

        private:
            T& l;
            subnet_wrapper<typename T::subnet_type,false> subnetwork;
        };
    }

// ----------------------------------------------------------------------------------------

    template <typename LAYER_DETAILS, typename SUBNET, typename enabled = void>
    class add_layer;

    template <typename T, typename U>
    struct is_nonloss_layer_type<add_layer<T,U>> : std::true_type {};

    template <typename LAYER_DETAILS, typename SUBNET>
    class add_layer<LAYER_DETAILS,SUBNET,
            typename std::enable_if<is_nonloss_layer_type<SUBNET>::value>::type>
    {
    public:
        typedef LAYER_DETAILS layer_details_type;
        typedef SUBNET subnet_type;
        typedef typename subnet_type::input_type input_type;
        const static size_t num_layers = subnet_type::num_layers + 1;
        const static size_t num_computational_layers = subnet_type::num_computational_layers + 1;

        add_layer(
        ):
            subnetwork(new subnet_type()),
            this_layer_setup_called(false),
            get_output_and_gradient_input_disabled(false)
        {
            if (this_layer_operates_inplace())
                subnetwork->disable_output_and_gradient_getters();
        }

        add_layer(const add_layer& item)
        {
            details = item.details;
            subnetwork.reset(new subnet_type(*item.subnetwork));
            this_layer_setup_called = item.this_layer_setup_called;
            get_output_and_gradient_input_disabled = item.get_output_and_gradient_input_disabled;
            cached_output = item.cached_output; 
        }
        add_layer& operator=(const add_layer& item) { add_layer(item).swap(*this); return *this;}
        add_layer(add_layer&& item) : add_layer() { swap(item); }
        add_layer& operator=(add_layer&& item) { swap(item); return *this; }

        template <typename T, typename U, typename E>
        friend class add_layer;
        template <typename T, bool is_first, typename E>
        friend class dimpl::subnet_wrapper;
        template <unsigned long T, typename U, typename E>
        friend class add_tag_layer;
        template <template<typename> class T, typename U>
        friend class add_skip_layer;
        template <size_t N, template<typename> class L, typename S>
        friend class repeat;

        // Allow copying networks from one to another as long as their corresponding 
        // layers can be constructed from each other.
        template <typename T, typename U, typename E>
        add_layer(
            const add_layer<T,U,E>& item
        ) :
            details(item.layer_details()), 
            subnetwork(new subnet_type(item.subnet())),
            this_layer_setup_called(item.this_layer_setup_called),
            get_output_and_gradient_input_disabled(item.get_output_and_gradient_input_disabled),
            cached_output(item.cached_output)
        {
            if (this_layer_operates_inplace())
                subnetwork->disable_output_and_gradient_getters();
        }

        template <typename ...T>
        add_layer(
            const LAYER_DETAILS& layer_det, 
            T&& ...args
        ) : 
            details(layer_det), 
            subnetwork(new subnet_type(std::forward<T>(args)...)),
            this_layer_setup_called(false),
            get_output_and_gradient_input_disabled(false)
        {
            if (this_layer_operates_inplace())
                subnetwork->disable_output_and_gradient_getters();
        }

        template <typename ...T>
        add_layer(
            LAYER_DETAILS&& layer_det, 
            T&& ...args
        ) : 
            details(std::move(layer_det)), 
            subnetwork(new subnet_type(std::forward<T>(args)...)),
            this_layer_setup_called(false),
            get_output_and_gradient_input_disabled(false)
        {
            if (this_layer_operates_inplace())
                subnetwork->disable_output_and_gradient_getters();
        }

        template <typename ...T, typename LD, typename ...U>
        add_layer(
            std::tuple<>,
            const std::tuple<LD,U...>& layer_det, 
            T&& ...args
        ) : add_layer(layer_det,args...) { }

        add_layer (
            std::tuple<>
        ) : add_layer() {}

        template <typename ...T>
        add_layer(
            std::tuple<>, 
            LAYER_DETAILS&& layer_det, 
            T&& ...args
        ) : add_layer(layer_det, args...) { }

        const tensor& forward(const tensor& x)
        {
            subnetwork->forward(x);
            const dimpl::subnet_wrapper<subnet_type> wsub(*subnetwork);
            if (!this_layer_setup_called)
            {
                throw std::runtime_error("layer params not loaded");
            }
            if (this_layer_operates_inplace())
                impl::call_layer_forward(details, wsub, private_get_output());
            else
                impl::call_layer_forward(details, wsub, cached_output);

            return private_get_output();
        }

    private:
        tensor& private_get_output() const
        { 
            if (const_cast<add_layer&>(*this).this_layer_operates_inplace())
                return subnetwork->private_get_output();
            else
                return const_cast<resizable_tensor&>(cached_output); 
        }
        void disable_output_and_gradient_getters (
        ) { get_output_and_gradient_input_disabled = true; }
    public:
        const tensor& get_output() const 
        { 
            if (get_output_and_gradient_input_disabled)
                throw dlib::error("Accessing this layer's get_output() is disabled because an in-place layer has been stacked on top of it.");
            return private_get_output(); 
        }

        const subnet_type& subnet() const { return *subnetwork; }
        subnet_type& subnet() { return *subnetwork; }

        const layer_details_type& layer_details() const { return details; } 
        layer_details_type& layer_details() { return details; } 

        void clean()
        {
            cached_output.clear();
            subnetwork->clean();
        }

        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {
            it = subnetwork->consume_params(it);
            it = details.consume_params(it);
            this_layer_setup_called = true;
            return it;
        }

        friend std::ostream& operator<< (std::ostream& out, const add_layer& item)
        {
            int min_length = 0;
            item.print(out, 0, min_length);
            return out;
        }

        void print (std::ostream& out, unsigned long idx, int& min_length) const
        {
            out << "layer<" << idx << ">\t" << impl::tensor_to_str(private_get_output(), min_length) << layer_details() << "\n";
            subnet().print(out, idx+1, min_length);
        }

    private:

        bool this_layer_operates_inplace(
        ) 
        {
            // This layer can run in-place if it's an in-place capable layer and also if
            // the layer it's on top of doesn't need its own output tensor (since in-place
            // layers overwrite that tensor)
            return impl::is_inplace_layer(details, *subnetwork) && !subnetwork->this_layer_requires_forward_output();
        }
        bool this_layer_requires_forward_output(
        ) 
        {
            return false;
        }

        void swap(add_layer& item)
        {
            std::swap(subnetwork,item.subnetwork);
            std::swap(details, item.details);
            std::swap(this_layer_setup_called, item.this_layer_setup_called);
            std::swap(get_output_and_gradient_input_disabled, item.get_output_and_gradient_input_disabled);
            std::swap(cached_output, item.cached_output);
        }


        LAYER_DETAILS details;
        std::unique_ptr<subnet_type> subnetwork;
        bool this_layer_setup_called;
        bool get_output_and_gradient_input_disabled;

        resizable_tensor cached_output; 
    };

    template <typename T, typename U, typename E>
    struct is_add_layer<add_layer<T,U,E>> : std::true_type {};
    template <typename T, typename U, typename E>
    struct is_add_layer<const add_layer<T,U,E>> : std::true_type {};
    template <typename T, typename U, typename E>
    struct is_add_layer<add_layer<T,U,E>&> : std::true_type {};
    template <typename T, typename U, typename E>
    struct is_add_layer<const add_layer<T,U,E>&> : std::true_type {};

// ----------------------------------------------------------------------------------------

// This version of add_layer handles the special case where the subnetwork being given is
// just an input layer object.
    template <typename LAYER_DETAILS, typename INPUT_LAYER, typename enabled>
    class add_layer
    {
    public:
        typedef LAYER_DETAILS layer_details_type;
        typedef INPUT_LAYER subnet_type;
        typedef typename INPUT_LAYER::input_type input_type;
        const static size_t num_layers = 2;
        const static size_t num_computational_layers = 1;

        add_layer(
        ): 
            this_layer_setup_called(false),
            get_output_and_gradient_input_disabled(false)
        {}

        add_layer(const add_layer&) = default;
        add_layer(add_layer&& item) : add_layer() { swap(item); }
        add_layer& operator=(const add_layer&) = default;
        add_layer& operator=(add_layer&& item) { swap(item); return *this; }

        template <typename T, typename U, typename E>
        friend class add_layer;
        template <typename T, bool is_first, typename E>
        friend class dimpl::subnet_wrapper;
        template <unsigned long T, typename U, typename E>
        friend class add_tag_layer;
        template <template<typename> class T, typename U>
        friend class add_skip_layer;
        template <size_t N, template<typename> class L, typename S>
        friend class repeat;

        // Allow copying networks from one to another as long as their corresponding 
        // layers can be constructed from each other.
        template <typename T, typename U, typename E>
        add_layer(
            const add_layer<T,U,E>& item
        ):
            input_layer(item.subnet()),
            details(item.layer_details()),
            this_layer_setup_called(item.this_layer_setup_called),
            get_output_and_gradient_input_disabled(false),
            cached_output(item.cached_output)
        {
        }

        add_layer(
            const LAYER_DETAILS& layer_det
        ) : 
            details(layer_det), 
            this_layer_setup_called(false),
            get_output_and_gradient_input_disabled(false)
        {}

        add_layer(
            const INPUT_LAYER& il 
        ) : 
            input_layer(il), 
            this_layer_setup_called(false),
            get_output_and_gradient_input_disabled(false)
        {}

        add_layer(
            LAYER_DETAILS&& layer_det
        ) : 
            details(std::move(layer_det)), 
            this_layer_setup_called(false),
            get_output_and_gradient_input_disabled(false)
        {}

        add_layer(
            LAYER_DETAILS layer_det, 
            INPUT_LAYER il
        ) : 
            details(std::move(layer_det)),
            input_layer(std::move(il)),
            this_layer_setup_called(false),
            get_output_and_gradient_input_disabled(false)
        {}

        add_layer(
            std::tuple<>,
            const LAYER_DETAILS& layer_det
        ) : add_layer(layer_det) {}

        add_layer(
            std::tuple<>,
            LAYER_DETAILS&& layer_det
        ) : add_layer(layer_det) {}

        add_layer(
            std::tuple<>,
            LAYER_DETAILS layer_det, 
            INPUT_LAYER il
        ) : add_layer(layer_det,il) {}

        const tensor& forward (const tensor& x)
        {
            subnet_wrapper wsub(x);
            if (!this_layer_setup_called)
            {
                throw std::runtime_error("layer params not loaded");
            }
            impl::call_layer_forward(details, wsub, cached_output);
            return private_get_output();
        }

    private:
        tensor& private_get_output() const { return const_cast<resizable_tensor&>(cached_output); }
        void disable_output_and_gradient_getters (
        ) { get_output_and_gradient_input_disabled = true; }
    public:
        const tensor& get_output() const 
        { 
            if (get_output_and_gradient_input_disabled)
                throw dlib::error("Accessing this layer's get_output() is disabled because an in-place layer has been stacked on top of it.");
            return private_get_output(); 
        }

        const subnet_type& subnet() const { return input_layer; } 
        subnet_type& subnet() { return input_layer; } 

        const layer_details_type& layer_details() const { return details; } 
        layer_details_type& layer_details() { return details; } 

        void clean()
        {
            cached_output.clear();
        }

        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {
            it = details.consume_params(it);
            this_layer_setup_called = true;
            return it;
        }

        friend std::ostream& operator<< (std::ostream& out, const add_layer& item)
        {
            int min_length = 0;
            item.print(out, 0, min_length);
            return out;
        }

        void print (std::ostream& out, unsigned long idx, int& min_length) const
        {
            out << "layer<" << idx << ">\t" << impl::tensor_to_str(private_get_output(), min_length) << layer_details() << "\n";

            // Don't print the repeat_input_layer since it doesn't exist from the user's
            // point of view.  It's just an artifact of how repeat<> works.
            if (!std::is_same<subnet_type, impl::repeat_input_layer>::value)
                out << "layer<" << idx+1 << ">\t" << subnet() << "\n";
        }

    private:

        bool this_layer_requires_forward_output(
        ) 
        {
            return false;
        }

        class subnet_wrapper
        {
        public:
            subnet_wrapper(const tensor& x_) :
                x(x_) {}

            subnet_wrapper(const subnet_wrapper&) = delete;
            subnet_wrapper& operator=(const subnet_wrapper&) = delete;

            const tensor& get_output() const { return x; }

        private:
            const tensor& x;
        };

        void swap(add_layer& item)
        {
            std::swap(input_layer, item.input_layer);
            std::swap(details, item.details);
            std::swap(this_layer_setup_called, item.this_layer_setup_called);
            std::swap(get_output_and_gradient_input_disabled, item.get_output_and_gradient_input_disabled);
            std::swap(cached_output, item.cached_output); 
        }

        subnet_type input_layer;
        LAYER_DETAILS details;
        bool this_layer_setup_called;
        bool get_output_and_gradient_input_disabled;
        resizable_tensor cached_output; 
    };

// ----------------------------------------------------------------------------------------

    template <unsigned long ID, typename SUBNET, typename enabled=void>
    class add_tag_layer;

    template <template<typename SUBNET> class tag>
    struct tag_id
    {
        const static unsigned long id = tag<impl::repeat_input_layer>::id;
    };

    template <unsigned long ID, typename SUBNET>
    class add_tag_layer<ID,SUBNET,
            typename std::enable_if<is_nonloss_layer_type<SUBNET>::value>::type>
    {
    public:
        typedef SUBNET subnet_type;
        typedef typename subnet_type::input_type input_type;
        typedef int layer_details_type; // not really used anywhere, but required by subnet_wrapper.
        const static size_t num_layers = subnet_type::num_layers + 1;
        const static size_t num_computational_layers = subnet_type::num_computational_layers;
        const static unsigned long id = ID;

        add_tag_layer() {};
        add_tag_layer(const add_tag_layer&) = default;
        add_tag_layer(add_tag_layer&&) = default;
        add_tag_layer& operator=(add_tag_layer&&) = default;
        add_tag_layer& operator=(const add_tag_layer&) = default;

        template <typename T>
        add_tag_layer(
            const add_tag_layer<ID,T>& item
        ) : subnetwork(item.subnet())
        {}

        template <typename ...T>
        add_tag_layer(
            T ...args
        ) : 
            subnetwork(std::move(args)...) 
        {
        }

        const tensor& forward(const tensor& x)
        {
            return subnetwork.forward(x);
        }

        const tensor& get_output() const { return subnetwork.get_output(); }

        const subnet_type& subnet() const { return subnetwork; }
        subnet_type& subnet() { return subnetwork; }

        void clean()
        {
            subnetwork.clean();
        }

        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {
            return subnetwork.consume_params(it);
        }

        friend std::ostream& operator<< (std::ostream& out, const add_tag_layer& item)
        {
            int min_length = 0;
            item.print(out, 0, min_length);
            return out;
        }

        void print (std::ostream& out, unsigned long idx, int& min_length) const
        {
            out << "layer<" << idx << ">\t" << impl::tensor_to_str(private_get_output(), min_length) << "tag" << ID << "\n";
            subnet().print(out, idx+1, min_length);
        }

    private:

        template <typename T, typename U, typename E>
        friend class add_layer;
        template <typename T, bool is_first, typename E>
        friend class dimpl::subnet_wrapper;
        template <unsigned long T, typename U, typename E>
        friend class add_tag_layer;
        template <template<typename> class T, typename U>
        friend class add_skip_layer;
        template <size_t N, template<typename> class L, typename S>
        friend class repeat;

        // You wouldn't put a tag on a layer if you didn't want to access its forward
        // outputs.  So this is always true.
        bool this_layer_requires_forward_output(
        ) { return true; } 

        void disable_output_and_gradient_getters (
        ) 
        { 
            // This should never happen because only inplace layers call
            // disable_output_and_gradient_getters(), however, putting a tag layer right
            // before an inplace layer basically means you don't want the following layer
            // to operate in place.  So the inplace layer should turn itself into an
            // out-of-place layer and not call disable_output_and_gradient_getters(). 
            DLIB_CASSERT(false,"This should never happen");
        }

        tensor& private_get_output() const
        { return subnetwork.private_get_output(); }

        subnet_type subnetwork;
    };

// ----------------------------------------------------------------------------------------

    template <typename ...T>
    struct decorator_repeat_group
    {
        decorator_repeat_group(
            T&& ...args
        ) : data(std::forward<T>(args)...) {}

        std::tuple<T...> data;
    };
    template <typename ...T>
    decorator_repeat_group<T...> repeat_group (
        T&& ...args
    )
    {
        return decorator_repeat_group<T...>(std::forward<T>(args)...);
    }

    template <
        size_t num,
        template<typename> class REPEATED_LAYER, 
        typename SUBNET
        >
    class repeat
    {
        static_assert(num > 0, "You can't have a layer repeated 0 times.");
    public:
        typedef SUBNET subnet_type;
        typedef typename SUBNET::input_type input_type;
        typedef int layer_details_type; // not really used anywhere, but required by subnet_wrapper.
        const static size_t comp_layers_in_each_group = (REPEATED_LAYER<SUBNET>::num_computational_layers-SUBNET::num_computational_layers);
        const static size_t comp_layers_in_repeated_group = comp_layers_in_each_group*num;
        const static size_t num_computational_layers = comp_layers_in_repeated_group + SUBNET::num_computational_layers;

        const static size_t layers_in_each_group = (REPEATED_LAYER<SUBNET>::num_layers-SUBNET::num_layers);
        const static size_t layers_in_repeated_group = layers_in_each_group*num;
        const static size_t num_layers = subnet_type::num_layers + layers_in_repeated_group;


        typedef REPEATED_LAYER<impl::repeat_input_layer> repeated_layer_type;

        repeat(
        ) : 
            details(num)
        {
        }

        size_t num_repetitions (
        ) const { return num; }

        const repeated_layer_type& get_repeated_layer (
            size_t i 
        ) const
        { 
            DLIB_CASSERT(i < num_repetitions());
            return details[i]; 
        }

        repeated_layer_type& get_repeated_layer (
            size_t i 
        ) 
        { 
            DLIB_CASSERT(i < num_repetitions());
            return details[i]; 
        }

        repeat(const repeat&) = default;
        repeat(repeat&&) = default;
        repeat& operator=(repeat&&) = default;
        repeat& operator=(const repeat&) = default;

        template <template<typename> class T, typename U>
        repeat(
            const repeat<num,T,U>& item
        ) : 
            subnetwork(item.subnetwork)
        {
            for (auto&& d : item.details)
                details.emplace_back(d);
        }

        template <typename T, typename ...U>
        repeat(
            T arg1,
            U ...args2
        ): 
            details(num, std::move(arg1)),
            subnetwork(std::move(args2)...)
        {
        }

        template <typename ...T, typename ...U>
        repeat(
            decorator_repeat_group<T...>&& arg1,
            U ...args2
        ): 
            details(num, arg1.data),
            subnetwork(std::move(args2)...)
        {
        }

        template <typename T, typename ...U>
        repeat(
            std::tuple<>,
            T arg1,
            U ...args2
        ): 
            details(num, std::move(arg1)),
            subnetwork(std::move(args2)...)
        {
        }

        const tensor& forward(const tensor& x)
        {
            subnetwork.forward(x);
            details[details.size()-1].forward(subnetwork.get_output());
            for (long i = details.size()-2; i >= 0; --i)
                details[i].forward(details[i+1].get_output());
            return private_get_output();
        }

    private:
        tensor& private_get_output() const
        { 
            return details[0].private_get_output();
        }
    public:
        const tensor& get_output() const 
        { 
            return details[0].get_output(); 
        }

        const subnet_type& subnet() const { return subnetwork; }
        subnet_type& subnet() { return subnetwork; }

        void clean()
        {
            subnetwork.clean();
            for (auto&& d : details)
                d.clean();
        }

        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {

            it = subnetwork.consume_params(it);
            it = details[details.size()-1].consume_params(it);
            for (long i = details.size()-2; i >= 0; --i)
                it = details[i].consume_params(it);

            return it;
        }

        friend std::ostream& operator<< (std::ostream& out, const repeat& item)
        {
            int min_length = 0;
            item.print(out, 0, min_length);
            return out;
        }

        void print (std::ostream& out, unsigned long idx, int& min_length) const
        {
            for (size_t i = 0; i < num_repetitions(); ++i)
            {
                get_repeated_layer(i).print(out, idx, min_length);
                idx += layers_in_each_group;
            }
            subnet().print(out, idx, min_length);
        }
    private:


        template <typename T, typename U, typename E>
        friend class add_layer;
        template <typename T, bool is_first, typename E>
        friend class dimpl::subnet_wrapper;
        template <unsigned long T, typename U, typename E>
        friend class add_tag_layer;
        template <template<typename> class T, typename U>
        friend class add_skip_layer;
        template <size_t N, template<typename> class L, typename S>
        friend class repeat;

        bool this_layer_requires_forward_output(
        ) 
        { 
            return details[0].this_layer_requires_forward_output(); 
        } 

        void disable_output_and_gradient_getters (
        ) 
        { 
            details[0].disable_output_and_gradient_getters();
        }


        std::vector<repeated_layer_type> details; 
        subnet_type subnetwork;
    };

    template <
        size_t num,
        template<typename> class REPEATED_LAYER, 
        typename SUBNET
        >
    struct is_nonloss_layer_type<repeat<num,REPEATED_LAYER,SUBNET>> : std::true_type {};

// ----------------------------------------------------------------------------------------

// This version of add_tag_layer handles the special case where the subnetwork being given
// is just an input layer object.
    template <unsigned long ID, typename INPUT_LAYER, typename enabled>
    class add_tag_layer
    {
    public:
        typedef INPUT_LAYER subnet_type;
        typedef typename subnet_type::input_type input_type;
        typedef int layer_details_type; // not really used anywhere, but required by subnet_wrapper.
        const static size_t num_computational_layers = 0;
        const static size_t num_layers = 2;
        const static unsigned long id = ID;

        add_tag_layer():cached_output_ptr(nullptr) {}

        add_tag_layer(const add_tag_layer&) = default;
        add_tag_layer& operator=(const add_tag_layer&) = default;
        add_tag_layer(add_tag_layer&& item) : add_tag_layer() { swap(item); }
        add_tag_layer& operator=(add_tag_layer&& item) { swap(item); return *this; }

        template <typename T, typename E>
        add_tag_layer(
            const add_tag_layer<ID,T,E>& item
        ) : input_layer(item.subnet()), 
            cached_output(item.cached_output),
            cached_output_ptr(nullptr)
        {}

        template <typename ...T>
        add_tag_layer(
            T ...args
        ) : 
            input_layer(std::move(args)...),
            cached_output_ptr(nullptr)
        {
        }

        add_tag_layer (
            std::tuple<>
        ) : 
            cached_output_ptr(nullptr)
        {}

        const tensor& operator() (const input_type& x)
        {
            return (*this)(&x, &x+1);
        }

        const tensor& forward(const tensor& x)
        {
            // If this tag is the first layer in one of the sub networks inside a repeat
            // layer then we don't want it to be creating copies of x.  This is because, we
            // can just hold a pointer to x since the way repeat is constructed guarantees
            // that x will have a lifetime larger than this pointer. 
            if (is_same_type<INPUT_LAYER, impl::repeat_input_layer>::value)
                cached_output_ptr = const_cast<tensor*>(&x);
            else
                cached_output = x;

            return get_output();
        }

        const tensor& get_output() const 
        { 
            if (cached_output_ptr)
                return *cached_output_ptr;
            else
                return cached_output; 
        }

        const subnet_type& subnet() const { return input_layer; }
        subnet_type& subnet() { return input_layer; }

        void clean()
        {
            cached_output.clear();
            cached_output_ptr = 0;
        }

        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {
            cached_output_ptr = nullptr;  
            return it;
        }

        friend std::ostream& operator<< (std::ostream& out, const add_tag_layer& item)
        {
            int min_length = 0;
            item.print(out, 0, min_length);
            return out;
        }

        void print (std::ostream& out, unsigned long idx, int& min_length) const
        {
            out << "layer<"<<idx << ">\t"<<impl::tensor_to_str(private_get_output(), min_length)<< "tag" << ID << "\n";
            // Don't print the repeat_input_layer since it doesn't exist from the user's
            // point of view.  It's just an artifact of how repeat<> works.
            if (!std::is_same<subnet_type, impl::repeat_input_layer>::value)
                out << "layer<"<< idx+1 << ">\t" << subnet() << "\n";
        }

    private:

        template <typename T, typename U, typename E>
        friend class add_layer;
        template <typename T, bool is_first, typename E>
        friend class dimpl::subnet_wrapper;
        template <unsigned long T, typename U, typename E>
        friend class add_tag_layer;
        template <template<typename> class T, typename U>
        friend class add_skip_layer;
        template <size_t N, template<typename> class L, typename S>
        friend class repeat;

        // You woudln't put a tag on a layer if you didn't want to access its forward
        // outputs.  So this is always true.
        bool this_layer_requires_forward_output(
        ) { return true; } 

        void disable_output_and_gradient_getters (
        ) 
        { 
            // This should never happen because only inplace layers call
            // disable_output_and_gradient_getters(), however, putting a tag layer right
            // before an inplace layer basically means you don't want the following layer
            // to operate in place.  So the inplace layer should turn itself into an
            // out-of-place layer and not call disable_output_and_gradient_getters(). 
            DLIB_CASSERT(false,"This should never happen");
        }

        tensor& private_get_output() const
        { return const_cast<tensor&>(get_output()); }

        void swap(add_tag_layer& item)
        {
            std::swap(input_layer, item.input_layer);
            std::swap(cached_output, item.cached_output);
            std::swap(cached_output_ptr, item.cached_output_ptr);
        }

        subnet_type input_layer;
        resizable_tensor cached_output;
        tensor* cached_output_ptr;
    };

    template <unsigned long ID, typename U, typename E>
    struct is_nonloss_layer_type<add_tag_layer<ID,U,E>> : std::true_type {};


// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    namespace impl
    {
        template <unsigned int i, typename T, typename enabled = void>
        struct layer_helper
        {
            static_assert(i < T::num_layers, "Call to layer() attempted to access non-existing layer in neural network.");
            static T& makeT();
            using next_type = typename std::remove_reference<decltype(makeT().subnet())>::type;
            using type = typename layer_helper<i-1,next_type>::type;
            static type& layer(T& n)
            {
                return layer_helper<i-1,next_type>::layer(n.subnet());
            }
        };
        template <
            unsigned int i,
            size_t N, template<typename> class L, typename S
        >
        struct layer_helper<i,repeat<N,L,S>, typename std::enable_if<(i!=0&&i>=repeat<N,L,S>::layers_in_repeated_group)>::type>
        {
            const static size_t layers_in_repeated_group = repeat<N,L,S>::layers_in_repeated_group;

            static repeat<N,L,S>& makeT();
            using next_type = typename std::remove_reference<decltype(makeT().subnet())>::type;
            using type = typename layer_helper<i-layers_in_repeated_group,next_type>::type;
            static type& layer(repeat<N,L,S>& n)
            {
                return layer_helper<i-layers_in_repeated_group,next_type>::layer(n.subnet());
            }
        };
        template <
            unsigned int i,
            size_t N, template<typename> class L, typename S
        >
        struct layer_helper<i,repeat<N,L,S>, typename std::enable_if<(i!=0&&i<repeat<N,L,S>::layers_in_repeated_group)>::type>
        {
            const static size_t layers_in_each_group = repeat<N,L,S>::layers_in_each_group;
            typedef typename repeat<N,L,S>::repeated_layer_type repeated_layer_type;
            using next_type = repeated_layer_type;
            using type = typename layer_helper<i%layers_in_each_group,next_type>::type;
            static type& layer(repeat<N,L,S>& n)
            {
                return layer_helper<i%layers_in_each_group,next_type>::layer(n.get_repeated_layer(i/layers_in_each_group));
            }
        };
        template <
            size_t N, template<typename> class L, typename S
        >
        struct layer_helper<0,repeat<N,L,S>, void>
        {
            typedef typename repeat<N,L,S>::repeated_layer_type repeated_layer_type;
            using type = repeated_layer_type;
            static type& layer(repeat<N,L,S>& n)
            {
                return n.get_repeated_layer(0);
            }
        };



        template <
            unsigned int i,
            size_t N, template<typename> class L, typename S
        >
        struct layer_helper<i,const repeat<N,L,S>, typename std::enable_if<(i!=0&&i>=repeat<N,L,S>::layers_in_repeated_group)>::type>
        {
            const static size_t layers_in_repeated_group = repeat<N,L,S>::layers_in_repeated_group;

            static const repeat<N,L,S>& makeT();
            using next_type = const typename std::remove_reference<decltype(makeT().subnet())>::type;
            using type = const typename layer_helper<i-layers_in_repeated_group,next_type>::type;
            static type& layer(const repeat<N,L,S>& n)
            {
                return layer_helper<i-layers_in_repeated_group,next_type>::layer(n.subnet());
            }
        };
        template <
            unsigned int i,
            size_t N, template<typename> class L, typename S
        >
        struct layer_helper<i,const repeat<N,L,S>, typename std::enable_if<(i!=0&&i<repeat<N,L,S>::layers_in_repeated_group)>::type>
        {
            const static size_t layers_in_each_group = repeat<N,L,S>::layers_in_each_group;
            typedef typename repeat<N,L,S>::repeated_layer_type repeated_layer_type;
            using next_type = const repeated_layer_type;
            using type = const typename layer_helper<i%layers_in_each_group,next_type>::type;
            static type& layer(const repeat<N,L,S>& n)
            {
                return layer_helper<i%layers_in_each_group,next_type>::layer(n.get_repeated_layer(i/layers_in_each_group));
            }
        };
        template <
            size_t N, template<typename> class L, typename S
        >
        struct layer_helper<0,const repeat<N,L,S>, void>
        {
            typedef typename repeat<N,L,S>::repeated_layer_type repeated_layer_type;
            using type = const repeated_layer_type;
            static type& layer(const repeat<N,L,S>& n)
            {
                return n.get_repeated_layer(0);
            }
        };



        template <typename T>
        struct layer_helper<0,T,void>
        {
            using type = T;
            static type& layer(T& n)
            {
                return n;
            }
        };

        template <template<typename> class Match, typename T, unsigned int i, typename enabled = void>
        struct layer_helper_match
        {
            static T& makeT();
            using next_type = typename std::remove_reference<decltype(makeT().subnet())>::type;
            using type = typename layer_helper_match<Match,next_type,i>::type;
            static type& layer(T& n)
            {
                return layer_helper_match<Match,next_type,i>::layer(n.subnet());
            }
        };
        // This overload catches add_layer and add_loss_layer templates.
        template <template<typename> class Match, typename T, unsigned int i>
        struct layer_helper_match<Match,T,i,
            typename std::enable_if<std::is_same<const T,const  Match<typename T::subnet_type>>::value>::type>
        {
            using type = typename layer_helper<i,T>::type;
            static type& layer(T& n)
            {
                return layer_helper<i,T>::layer(n);
            }
        };
        // This overload catches input templates.
        template <template<typename> class Match, typename T, unsigned int i>
        struct layer_helper_match<Match,T,i,
            typename std::enable_if<std::is_same<const T,const  Match<typename T::input_type>>::value>::type>
        {
            using type = typename layer_helper<i,T>::type;
            static type& layer(T& n)
            {
                return layer_helper<i,T>::layer(n);
            }
        };
        // This overload catches subnet_wrapper templates.
        template <template<typename> class Match, typename T, unsigned int i>
        struct layer_helper_match<Match,T,i,
            typename std::enable_if<std::is_same<const typename T::wrapped_type, 
                                                 const Match<typename T::wrapped_type::subnet_type>>::value>::type>
        {
            using type = typename layer_helper<i,T>::type;
            static type& layer(T& n)
            {
                return layer_helper<i,T>::layer(n);
            }
        };
    }

    template <unsigned int i, typename T>
    typename impl::layer_helper<i,T>::type& layer (T& n) 
    {
        return impl::layer_helper<i,T>::layer(n);
    }

    template <template<typename> class Match, typename T>
    typename impl::layer_helper_match<Match,T,0>::type& layer (T& n) 
    {
        return impl::layer_helper_match<Match,T,0>::layer(n);
    }

    template <template<typename> class Match, unsigned int i, typename T>
    typename impl::layer_helper_match<Match,T,i>::type& layer (T& n) 
    {
        return impl::layer_helper_match<Match,T,i>::layer(n);
    }

// ----------------------------------------------------------------------------------------


    namespace dimpl
    {
        template <typename T>
        T& get_input_details (
            T& net
        ) 
        { 
            return net; 
        } 

        template <typename T, bool is_first, typename enabled>
        auto get_input_details (
            dimpl::subnet_wrapper<T,is_first,enabled>& net
        ) -> decltype(net.layer_details())&
        {
            return net.layer_details();
        }

        template <typename T, bool is_first, typename enabled>
        auto get_input_details (
            const dimpl::subnet_wrapper<T,is_first,enabled>& net
        ) -> decltype(net.layer_details())&
        {
            return net.layer_details();
        }
    }

    template <typename net_type>
    auto input_layer (
        net_type& net
    ) -> decltype(dimpl::get_input_details(layer<net_type::num_layers-1>(net)))&
    {
        // Calling input_layer() on a subnet_wrapper is a little funny since the behavior of
        // .subnet() returns another subnet_wrapper rather than an input details object as it
        // does in add_layer.
        return dimpl::get_input_details(layer<net_type::num_layers-1>(net));
    }

// ----------------------------------------------------------------------------------------

    template <template<typename> class TAG_TYPE, typename SUBNET>
    class add_skip_layer
    {
    public:
        typedef SUBNET subnet_type;
        typedef typename subnet_type::input_type input_type;
        typedef int layer_details_type; // not really used anywhere, but required by subnet_wrapper.
        const static size_t num_layers = subnet_type::num_layers + 1;
        const static size_t num_computational_layers = subnet_type::num_computational_layers;
        const static unsigned long id = tag_id<TAG_TYPE>::id;

        add_skip_layer() {};
        add_skip_layer(const add_skip_layer&) = default;
        add_skip_layer(add_skip_layer&&) = default;
        add_skip_layer& operator=(add_skip_layer&&) = default;
        add_skip_layer& operator=(const add_skip_layer&) = default;

        template <typename T>
        add_skip_layer(
            const add_skip_layer<TAG_TYPE,T>& item
        ) : subnetwork(item.subnet())
        {}

        template <typename ...T>
        add_skip_layer(
            T ...args
        ) : 
            subnetwork(std::move(args)...) 
        {
        }

        template <typename forward_iterator>
        const tensor& operator() (
            forward_iterator ibegin,
            forward_iterator iend
        )
        {
            subnetwork(ibegin,iend);
            return layer<TAG_TYPE>(subnetwork).get_output();
        }

        const tensor& operator() (const input_type& x)
        {
            subnetwork(x);
            return layer<TAG_TYPE>(subnetwork).get_output();
        }

        const tensor& forward(const tensor& x)
        {
            subnetwork.forward(x);
            return layer<TAG_TYPE>(subnetwork).get_output();
        }

        const tensor& get_output() const 
        { 
            return layer<TAG_TYPE>(subnetwork).get_output();
        }

        const subnet_type& subnet() const 
        { 
            return subnetwork; 
        }

        subnet_type& subnet() 
        { 
            return subnetwork; 
        }

        void clean()
        {
            subnetwork.clean();
        }

        std::vector<param_data>::const_iterator 
        consume_params(std::vector<param_data>::const_iterator it) {
            return subnetwork.consume_params(it);  
        }

        friend std::ostream& operator<< (std::ostream& out, const add_skip_layer& item)
        {
            int min_length = 0;
            item.print(out, 0, min_length);
            return out;
        }

        void print (std::ostream& out, unsigned long idx, int& min_length) const
        {
            out << "layer<" << idx << ">\t"<<impl::tensor_to_str(private_get_output(), min_length) <<"skip"<<id<<"\n";
            subnet().print(out, idx+1, min_length);
        }

    private:


        template <typename T, typename U, typename E>
        friend class add_layer;
        template <typename T, bool is_first, typename E>
        friend class dimpl::subnet_wrapper;
        template <unsigned long T, typename U, typename E>
        friend class add_tag_layer;
        template <template<typename> class T, typename U>
        friend class add_skip_layer;
        template <size_t N, template<typename> class L, typename S>
        friend class repeat;

        bool this_layer_requires_forward_output(
        ) { return layer<TAG_TYPE>(subnetwork).this_layer_requires_forward_output(); } 

        void disable_output_and_gradient_getters (
        ) { layer<TAG_TYPE>(subnetwork).disable_output_and_gradient_getters(); }

        tensor& private_get_output() const
        { return layer<TAG_TYPE>(subnetwork).private_get_output(); }

        subnet_type subnetwork;
    };
    template <template<typename> class T, typename U>
    struct is_nonloss_layer_type<add_skip_layer<T,U>> : std::true_type {};

    template <typename SUBNET> using tag1  = add_tag_layer< 1, SUBNET>;
    template <typename SUBNET> using tag2  = add_tag_layer< 2, SUBNET>;
    template <typename SUBNET> using tag3  = add_tag_layer< 3, SUBNET>;
    template <typename SUBNET> using tag4  = add_tag_layer< 4, SUBNET>;
    template <typename SUBNET> using tag5  = add_tag_layer< 5, SUBNET>;
    template <typename SUBNET> using tag6  = add_tag_layer< 6, SUBNET>;
    template <typename SUBNET> using tag7  = add_tag_layer< 7, SUBNET>;
    template <typename SUBNET> using tag8  = add_tag_layer< 8, SUBNET>;
    template <typename SUBNET> using tag9  = add_tag_layer< 9, SUBNET>;
    template <typename SUBNET> using tag10 = add_tag_layer<10, SUBNET>;

    template <typename SUBNET> using skip1  = add_skip_layer< tag1, SUBNET>;
    template <typename SUBNET> using skip2  = add_skip_layer< tag2, SUBNET>;
    template <typename SUBNET> using skip3  = add_skip_layer< tag3, SUBNET>;
    template <typename SUBNET> using skip4  = add_skip_layer< tag4, SUBNET>;
    template <typename SUBNET> using skip5  = add_skip_layer< tag5, SUBNET>;
    template <typename SUBNET> using skip6  = add_skip_layer< tag6, SUBNET>;
    template <typename SUBNET> using skip7  = add_skip_layer< tag7, SUBNET>;
    template <typename SUBNET> using skip8  = add_skip_layer< tag8, SUBNET>;
    template <typename SUBNET> using skip9  = add_skip_layer< tag9, SUBNET>;
    template <typename SUBNET> using skip10 = add_skip_layer<tag10, SUBNET>;


// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_CORE_H_


