#pragma once
#include "config.h"
#include "tensor.h"
#include "common.h"

using namespace std;

class Params
{
public:
    vector<std::pair<string, TensorObj>> params_;

    void addParam(const string &name, const TensorObj &value)
    {
        params_.emplace_back(name, value);
        // params_[name] = value;
    }

    TensorObj &operator[](const std::string &name)
    {
        auto it = find_if(params_.begin(), params_.end(),
                          [&name](const std::pair<std::string, TensorObj> &pair)
                          {
                              return pair.first == name;
                          });
        if (it != params_.end())
        {
            return it->second;
        }
        else
        {
            throw std::out_of_range("Param not found: " + name);
        }
    }
};

class Activations
{
public:
    vector<std::pair<string, TensorObj>> acts_;

    void addAct(const string &name, const TensorObj &value)
    {
        acts_.emplace_back(name, value);
        // acts_[name] = value;
    }

    TensorObj &operator[](const std::string &name)
    {
        auto it = find_if(acts_.begin(), acts_.end(),
                          [&name](const std::pair<std::string, TensorObj> &pair)
                          {
                              return pair.first == name;
                          });
        if (it != acts_.end())
        {
            return it->second;
        }
        else
        {
            throw std::out_of_range("Acts not found: " + name);
        }
    }
};

class Net
{
public:
    ModelConfig config;
    int num_params;
    int num_acts = 0;
    Params params;
    Activations acts;
    Params param_grads;
    Activations act_grads;
    float mean_loss = -1.0f;

    Net(const string &checkpoint_path) noexcept;
    void forward(int *inputs, int *targets, int B, int T);
    void backward(int *inputs, int *targets, int B, int T);
};
