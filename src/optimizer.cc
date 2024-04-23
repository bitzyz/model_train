#include "optimizer.h"

void AdamwOptimizer::zero_grad()
{
    if (net->param_grads.params_.size() == 0)
    {
        for (auto &param : net->params.params_)
        {
            auto name = param.first;
            pair<string, TensorObj> t = {name, TensorObj(param.second.shape)};
            int N = t.second.size();
            float *grad_param = reinterpret_cast<float *>(t.second.ptr);
            for (int i = 0; i < N; i++)
            {
                grad_param[i] = 0.0f;
            }
            net->param_grads.params_.push_back(t);
        }
    }
    else
    {
        for (auto &param : net->param_grads.params_)
        {
            int N = param.second.size();
            float *grad_param = reinterpret_cast<float *>(param.second.ptr);
            for (int i = 0; i < N; i++)
            {
                grad_param[i] = 0.0f;
            }
        }
    }
    if (net->act_grads.acts_.size() == 0)
    {
        for (auto &act : net->acts.acts_)
        {
            auto name = act.first;
            pair<string, TensorObj> t = {name, TensorObj(act.second.shape)};
            int N = t.second.size();
            float *grad_act = reinterpret_cast<float *>(t.second.ptr);
            for (int i = 0; i < N; i++)
            {
                grad_act[i] = 0.0f;
            }
            net->act_grads.acts_.push_back(t);
        }
    }
    else
    {
        for (auto &act : net->act_grads.acts_)
        {
            int N = act.second.size();
            float *grad_act = reinterpret_cast<float *>(act.second.ptr);
            for (int i = 0; i < N; i++)
            {
                grad_act[i] = 0.0f;
            }
        }
    }
}

void AdamwOptimizer::update(int t)
{
    if (!m_memory)
    {
        m_memory = new float[net->num_params];
        fill_n(m_memory, net->num_params, 0.0f);
        v_memory = new float[net->num_params];
        fill_n(v_memory, net->num_params, 0.0f);
    }
    int count = 0;
    for (int i = 0; i < net->params.params_.size(); i++)
    {
        if (count >= net->num_params)
        {
            cerr << "Error: count exceeds the number of parameters" << endl;
            exit(1);
        }
        auto param = net->params.params_[i];
        auto grad = net->param_grads.params_[i];
        int N = param.second.size();
        float *param_val = reinterpret_cast<float *>(param.second.ptr);
        float *grad_val = reinterpret_cast<float *>(grad.second.ptr);
        for (int i = 0; i < N; i++)
        {
            float m = beta1 * m_memory[count + i] + (1.0f - beta1) * grad_val[i];
            float v = beta2 * v_memory[count + i] + (1.0f - beta2) * grad_val[i] * grad_val[i];
            float m_hat = m / (1.0f - pow(beta1, t));
            float v_hat = v / (1.0f - pow(beta2, t));

            // update
            m_memory[count + i] = m;
            v_memory[count + i] = v;
            param_val[i] -= learning_rate * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * param_val[i]);
        }
        count += N;
    }
}