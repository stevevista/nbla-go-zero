#pragma once
#include <memory>
#include <engine/tensor.h>
#include <engine/leela/Interface.h>

// For windows support
#if defined(_MSC_VER) && !defined(__CUDACC__)
#ifdef simplemodel_EXPORTS
#define MODEL_API __declspec(dllexport)
#else
#define MODEL_API __declspec(dllimport)
#endif
#else
#define MODEL_API
#endif



using std::shared_ptr;
using std::vector;
using std::pair;

class MODEL_API ZeroPredictModel : public IPreditModel, public ILeelaModel  {
public:
    static constexpr int boardsize = 19;
    static constexpr int residual_layers = 19;

    ZeroPredictModel();

    bool load_weights(const std::string& path);

    typedef vector<float> distribution_t;

    virtual pair<vector<float>, float> predict(const InputFeature& features, float temperature);
    virtual vector<vector<float>> predict_distribution(const InputTensor& planes, float temprature);
	virtual vector<float> predict_winrate(const InputTensor& planes);

private:
    class ModelImpl;
    std::shared_ptr<ModelImpl> impl_;
};
