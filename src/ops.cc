#include "ops.h"

EncoderOp::EncoderOp(vector<TensorObj> inputs_, vector<TensorObj> outputs_) : Ops(inputs_, outputs_) {}

void EncoderOp::forward()
{
    int B = inputs[0].shape[0];
    int T = inputs[0].shape[1];
    int C = inputs[1].shape[1];
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            // seek to the output position in out[b,t,:]
            float *out_bt = reinterpret_cast<float *>(outputs[0].ptr) + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            int *ix = reinterpret_cast<int *>(inputs[0].ptr) + b * T + t;
            // seek to the position in wte corresponding to the token
            float *wte_ix = reinterpret_cast<float *>(inputs[1].ptr) + (*ix) * C;
            // seek to the position in wpe corresponding to the position
            float *wpe_t = reinterpret_cast<float *>(inputs[2].ptr) + t * C;
            // add the two vectors and store the result in out[b,t,:]
            for (int c = 0; c < C; c++)
            {
                out_bt[c] = wte_ix[c] + wpe_t[c];
            }
        }
    }
}

void EncoderOp::backward()
{
    // inputs[0] is dout (B,T,C)
    // inputs[1] is model inputs
    // outputs[0] is grad_wte
    // outputs[1] is grad_wpe

    int B = inputs[0].shape[0];
    int T = inputs[0].shape[1];
    int C = inputs[0].shape[2];

    for (int b = 0; b < B; b++)
    {
        int *ix = reinterpret_cast<int *>(inputs[1].ptr) + b * T;
        for (int t = 0; t < T; t++)
        {
            float *dout_bt = reinterpret_cast<float *>(inputs[0].ptr) + b * T * C + t * C;
            int ix_val = ix[t];
            float *dwte_ix = reinterpret_cast<float *>(outputs[0].ptr) + ix_val * C;
            float *dwpe_t = reinterpret_cast<float *>(outputs[1].ptr) + t * C;
            for (int c = 0; c < C; c++)
            {
                float d = dout_bt[c];
                dwte_ix[c] += d;
                dwpe_t[c] += d;
            }
        }
    }
}

LayerNormOp::LayerNormOp(vector<TensorObj> inputs_, vector<TensorObj> outputs_) : Ops(inputs_, outputs_) {}

void LayerNormOp::forward()
{
    // inputs[0] is inp (B,T,C)
    // inputs[1] is weight (C,)
    // inputs[2] is bias (C,)
    // outputs[0] is out (B,T,C)
    // outputs[1] is mean (B,T)
    // outputs[2] is rstd (B,T)
    int B = inputs[0].shape[0];
    int T = inputs[0].shape[1];
    int C = inputs[0].shape[2];

    float eps = 1e-5f;
    for (int b = 0; b < B; b++)
    {
        float *mean = reinterpret_cast<float *>(outputs[1].ptr) + b * T;
        float *rstd = reinterpret_cast<float *>(outputs[2].ptr) + b * T;
        for (int t = 0; t < T; t++)
        {
            // seek to the input position inp[b,t,:]
            float *x = reinterpret_cast<float *>(inputs[0].ptr) + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int c = 0; c < C; c++)
            {
                m += x[c];
            }
            m /= C;
            // calculate the variance
            float v = 0.0f;
            for (int c = 0; c < C; c++)
            {
                v += (x[c] - m) * (x[c] - m);
            }
            v /= C;
            // calculate the standard deviation
            float s = 1.0f / sqrt(v + eps);
            // seek to the output position out[b,t,:]
            float *out_bt = reinterpret_cast<float *>(outputs[0].ptr) + b * T * C + t * C;
            // normalize the input and store the result in out[b,t,:]
            for (int c = 0; c < C; c++)
            {
                out_bt[c] = (x[c] - m) * s;
                // out * weight + bias
                out_bt[c] = out_bt[c] * reinterpret_cast<float *>(inputs[1].ptr)[c] + reinterpret_cast<float *>(inputs[2].ptr)[c];
            }
            mean[t] = m;
            rstd[t] = s;
        }
    }
}

void LayerNormOp::backward()
{
    // inputs[0] is dout (B,T,C)
    // inputs[1] is inp  (B,T,C)
    // inputs[2] is weight
    // inputs[3] is mean
    // inputs[4] is rstd
    // outputs[0] is grad_inp
    // outputs[1] is grad_weight
    // outputs[2] is grad_bias

    int B = inputs[0].shape[0];
    int T = inputs[0].shape[1];
    int C = inputs[0].shape[2];

    for (int b = 0; b < B; b++)
    {
        float *mean = reinterpret_cast<float *>(inputs[3].ptr) + b * T;
        float *rstd = reinterpret_cast<float *>(inputs[4].ptr) + b * T;
        for (int t = 0; t < T; t++)
        {
            float *dout_bt = reinterpret_cast<float *>(inputs[0].ptr) + b * T * C + t * C;
            float *inp_bt = reinterpret_cast<float *>(inputs[1].ptr) + b * T * C + t * C;
            float *dinp_bt = reinterpret_cast<float *>(outputs[0].ptr) + b * T * C + t * C;
            float mean_t = mean[t];
            float rstd_t = rstd[t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int c = 0; c < C; c++)
            {
                float norm_bti = (inp_bt[c] - mean_t) * rstd_t;
                float dnorm_i = reinterpret_cast<float *>(inputs[2].ptr)[c] * dout_bt[c];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += norm_bti * dnorm_i;
            }
            dnorm_mean /= C;
            dnorm_norm_mean /= C;
            // now iterate again and accumulate all the gradients
            for (int c = 0; c < C; c++)
            {
                float norm_bti = (inp_bt[c] - mean_t) * rstd_t;
                float dnorm_i = reinterpret_cast<float *>(inputs[2].ptr)[c] * dout_bt[c];
                // gradient contribution to bias
                reinterpret_cast<float *>(outputs[2].ptr)[c] += dnorm_i;
                // gradient contribution to weights
                reinterpret_cast<float *>(outputs[1].ptr)[c] += norm_bti * dout_bt[c];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i;
                dval -= dnorm_mean;
                dval -= dnorm_norm_mean * norm_bti;
                dval *= rstd_t;
                dinp_bt[c] += dval;
            }
        }
    }
}

MatMulOp::MatMulOp(vector<TensorObj> inputs_, vector<TensorObj> outputs_) : Ops(inputs_, outputs_) {}

void MatMulOp::forward()
{
    // inputs[0] is inp
    // inputs[1] is weight
    // inputs[2] is bias
    // outputs[0] is out
    int B = inputs[0].shape[0];
    int T = inputs[0].shape[1];
    int C = inputs[0].shape[2];
    int OC = inputs[1].shape[0];
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            // seek to the output position in out[b,t,:]
            float *out_bt = reinterpret_cast<float *>(outputs[0].ptr) + b * T * OC + t * OC;
            // seek to the input position in inp[b,t,:]
            float *inp_bt = reinterpret_cast<float *>(inputs[0].ptr) + b * T * C + t * C;
            for (int o = 0; o < OC; o++)
            {
                float val = inputs.size() == 2 ? 0.0f : reinterpret_cast<float *>(inputs[2].ptr)[o];
                float *wrow = reinterpret_cast<float *>(inputs[1].ptr) + o * C;
                for (int c = 0; c < C; c++)
                {
                    val += inp_bt[c] * wrow[c];
                }
                out_bt[o] = val;
            }
        }
    }
}

void MatMulOp::backward()
{
    // inputs[0] is dout (B,T,OC)
    // inputs[1] is inp  (B,T,C)
    // inputs[2] is weight
    // outputs[0] is grad_inp
    // outputs[1] is grad_weight
    // outputs[2] is grad_bias

    int B = inputs[0].shape[0];
    int T = inputs[0].shape[1];
    int OC = inputs[0].shape[2];
    int C = inputs[1].shape[2];

// backward into inp first
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            float *dout_bt = reinterpret_cast<float *>(inputs[0].ptr) + b * T * OC + t * OC;
            float *dinp_bt = reinterpret_cast<float *>(outputs[0].ptr) + b * T * C + t * C;

            for (int o = 0; o < OC; o++)
            {
                float *wrow = reinterpret_cast<float *>(inputs[2].ptr) + o * C;
                float d = dout_bt[o];
                for (int c = 0; c < C; c++)
                {
                    dinp_bt[c] += d * wrow[c];
                }
            }
        }
    }

// backward into weight/bias
#pragma omp parallel for
    for (int o = 0; o < OC; o++)
    {
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float *dout_bt = reinterpret_cast<float *>(inputs[0].ptr) + b * T * OC + t * OC;
                float *inp_bt = reinterpret_cast<float *>(inputs[1].ptr) + b * T * C + t * C;
                float *dwrow = reinterpret_cast<float *>(outputs[1].ptr) + o * C;
                float d = dout_bt[o];
                // have bias
                if (outputs.size() == 3)
                {
                    float *dbias = reinterpret_cast<float *>(outputs[2].ptr);
                    dbias[o] += d;
                }
                for (int i = 0; i < C; i++)
                {
                    dwrow[i] += d * inp_bt[i];
                }
            }
        }
    }
}

AttentionOp::AttentionOp(vector<TensorObj> inputs_, vector<TensorObj> outputs_) : Ops(inputs_, outputs_) {}

void AttentionOp::forward()
{
    // inputs[0] is qkv (B,T,3C)
    // outputs[0] is atty (B,T,C)
    // outputs[1] is preatt (B,NH,T,T)
    // outputs[2] is att (B,NH,T,T)
    int B = inputs[0].shape[0];
    int T = inputs[0].shape[1];
    int C3 = inputs[0].shape[2];
    int C = outputs[0].shape[2];
    int NH = outputs[1].shape[1];
    int hs = C / NH;
    float scale = 1.0f / sqrt(hs);
#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < NH; h++)
            {
                float *query_t = reinterpret_cast<float *>(inputs[0].ptr) + b * T * C3 + t * C3 + h * hs;
                float *preatt_bth = reinterpret_cast<float *>(outputs[1].ptr) + b * NH * T * T + h * T * T + t * T;
                float *att_bth = reinterpret_cast<float *>(outputs[2].ptr) + b * NH * T * T + h * T * T + t * T;

                // 1. calculate the attention scores (QKT)
                float maxval = -10000.0f;
                for (int t2 = 0; t2 <= t; t2++)
                {
                    float *key_t2 = reinterpret_cast<float *>(inputs[0].ptr) + b * T * C3 + t2 * C3 + h * hs + C;
                    // query_t * key_t2
                    float val = 0.0f;
                    for (int c = 0; c < hs; c++)
                    {
                        val += query_t[c] * key_t2[c];
                    }
                    val *= scale;
                    maxval = max(maxval, val);
                    preatt_bth[t2] = val;
                }
                // 2. calculate the exp and sum
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++)
                {
                    float expv = exp(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
                // 3. normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++)
                {
                    if (t2 <= t)
                    {
                        att_bth[t2] *= expsum_inv;
                    }
                    else
                    {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }
                // 4. calculate the weighted values into the output of attention
                float *out_bth = reinterpret_cast<float *>(outputs[0].ptr) + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++)
                {
                    out_bth[i] = 0.0f;
                }
                for (int t2 = 0; t2 <= t; t2++)
                {
                    float *value_t2 = reinterpret_cast<float *>(inputs[0].ptr) + b * T * C3 + t2 * C3 + h * hs + 2 * C;
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++)
                    {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

void AttentionOp::backward()
{
    // inputs[0] is dout (B,T,C)
    // inputs[1] is qkv  (B,T,3C)
    // inputs[2] is att (B,NH,T,T)
    // outputs[0] is grad_qkv
    // outputs[1] is grad_preatt
    // outputs[2] is grad_att

    int B = inputs[0].shape[0];
    int T = inputs[0].shape[1];
    int C = inputs[0].shape[2];
    int NH = inputs[2].shape[1];
    int C3 = inputs[1].shape[2];
    int hs = C / NH;
    float scale = 1.0f / sqrt(hs);

    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < NH; h++)
            {
                float *att_bth = reinterpret_cast<float *>(inputs[2].ptr) + b * NH * T * T + h * T * T + t * T;
                float *datt_bth = reinterpret_cast<float *>(outputs[2].ptr) + b * NH * T * T + h * T * T + t * T;
                float *dpreatt_bth = reinterpret_cast<float *>(outputs[1].ptr) + b * NH * T * T + h * T * T + t * T;
                float *dquery_t = reinterpret_cast<float *>(outputs[0].ptr) + b * T * C3 + t * C3 + h * hs;
                float *query_t = reinterpret_cast<float *>(inputs[1].ptr) + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                float *dout_bth = reinterpret_cast<float *>(inputs[0].ptr) + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 <= t; t2++)
                {
                    float *value_t2 = reinterpret_cast<float *>(inputs[1].ptr) + b * T * C3 + t2 * C3 + h * hs + 2 * C;
                    float *dvalue_t2 = reinterpret_cast<float *>(outputs[0].ptr) + b * T * C3 + t2 * C3 + h * hs + 2 * C;
                    for (int i = 0; i < hs; i++)
                    {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }
                // backward pass 2&3, the softmax
                for (int t2 = 0; t2 <= t; t2++)
                {
                    for (int t3 = 0; t3 <= t; t3++)
                    {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++)
                {
                    float *key_t2 = reinterpret_cast<float *>(inputs[1].ptr) + b * T * C3 + t2 * C3 + h * hs + C;
                    float *dkey_t2 = reinterpret_cast<float *>(outputs[0].ptr) + b * T * C3 + t2 * C3 + h * hs + C;
                    for (int i = 0; i < hs; i++)
                    {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

ResidualOp::ResidualOp(vector<TensorObj> inputs_, vector<TensorObj> outputs_) : Ops(inputs_, outputs_) {}

void ResidualOp::forward()
{
    int B = inputs[0].shape[0];
    int T = inputs[0].shape[1];
    int C = inputs[0].shape[2];

    int N = B * T * C;
    float *inp = reinterpret_cast<float *>(inputs[0].ptr);
    float *inp2 = reinterpret_cast<float *>(inputs[1].ptr);
    float *out = reinterpret_cast<float *>(outputs[0].ptr);
    for (int i = 0; i < N; i++)
    {
        out[i] = inp[i] + inp2[i];
    }
}

void ResidualOp::backward()
{
    // inputs[0] shape is (B,T,C)
    // outputs[0] shape is (B,T,C)
    // outputs[1] shape is (B,T,C)

    int B = inputs[0].shape[0];
    int T = inputs[0].shape[1];
    int C = inputs[0].shape[2];

    int N = B * T * C;
    float *grad_inp = reinterpret_cast<float *>(inputs[0].ptr);
    float *grad_out = reinterpret_cast<float *>(outputs[0].ptr);
    float *grad_out2 = reinterpret_cast<float *>(outputs[1].ptr);
    for (int i = 0; i < N; i++)
    {
        grad_out[i] += grad_inp[i];
        grad_out2[i] += grad_inp[i];
    }
}

GeluOp::GeluOp(vector<TensorObj> inputs_, vector<TensorObj> outputs_) : Ops(inputs_, outputs_) {}

void GeluOp::forward()
{
    // inputs[0] shape is (B,T,4*C)
    int B = inputs[0].shape[0];
    int T = inputs[0].shape[1];
    int C = inputs[0].shape[2];

    int N = B * T * C;
    float *inp = reinterpret_cast<float *>(inputs[0].ptr);
    float *out = reinterpret_cast<float *>(outputs[0].ptr);
    float s = sqrt(2.0f / 3.14159265358979323846f);
    for (int i = 0; i < N; i++)
    {
        float x = inp[i];
        out[i] = 0.5f * x * (1.0f + tanh(s * (x + 0.044715f * x * x * x)));
    }
}

void GeluOp::backward()
{
    // inputs[0] is grad_act[fch_gelu],shape is (B,T,4*C)
    // inputs[1] is act[fch],shape is (B,T,4*C)
    // output[0] is grad_act[fch],shape is (B,T,4*C)

    int B = inputs[0].shape[0];
    int T = inputs[0].shape[1];
    int C = inputs[0].shape[2];

    int N = B * T * C;
    float s = sqrt(2.0f / 3.14159265358979323846f);
    float *inp_grad = reinterpret_cast<float *>(inputs[0].ptr);
    float *inp_act = reinterpret_cast<float *>(inputs[1].ptr);
    float *out_grad = reinterpret_cast<float *>(outputs[0].ptr);
    for (int i = 0; i < N; i++)
    {
        float x = inp_act[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = s * (x + cube);
        float tanh_out = tanh(tanh_arg);
        float coshf_out = cosh(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * s * (1.0f + 3.0f * 0.044715f * x * x);
        out_grad[i] += local_grad * inp_grad[i];
    }
}

SoftmaxOp::SoftmaxOp(vector<TensorObj> inputs_, vector<TensorObj> outputs_) : Ops(inputs_, outputs_) {}

void SoftmaxOp::forward()
{
    // inputs[0] is logits is (B,T,V)
    // outputs[0] is probs is (B,T,V)
    int B = inputs[0].shape[0];
    int T = inputs[0].shape[1];
    int V = inputs[0].shape[2];
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            float *logits_bt = reinterpret_cast<float *>(inputs[0].ptr) + b * T * V + t * V;
            float *probs_bt = reinterpret_cast<float *>(outputs[0].ptr) + b * T * V + t * V;
            float maxval = -10000.0f;
            for (int v = 0; v < V; v++)
            {
                maxval = max(maxval, logits_bt[v]);
            }
            float sum = 0.0f;
            for (int v = 0; v < V; v++)
            {
                float val = exp(logits_bt[v] - maxval);
                probs_bt[v] = val;
                sum += val;
            }
            for (int v = 0; v < V; v++)
            {
                probs_bt[v] /= sum;
            }
        }
    }
}

void SoftmaxOp::backward()
{
    // wait to implement
}

CrossEntropyOp::CrossEntropyOp(vector<TensorObj> inputs_, vector<TensorObj> outputs_) : Ops(inputs_, outputs_) {}

void CrossEntropyOp::forward()
{
    // inputs[0] is probs (B,T,V)
    // inputs[1] is target (B,T)
    // outputs[0] is loss (B,T)
    int B = inputs[0].shape[0];
    int T = inputs[0].shape[1];
    int V = inputs[0].shape[2];

    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            float *probs_bt = reinterpret_cast<float *>(inputs[0].ptr) + b * T * V + t * V;
            int *target_bt = reinterpret_cast<int *>(inputs[1].ptr) + b * T + t;
            float *loss_bt = reinterpret_cast<float *>(outputs[0].ptr) + b * T + t;
            *loss_bt = -log(probs_bt[*target_bt]);
        }
    }
}

void CrossEntropyOp::backward()
{
    // input[0] is grad_act[losses], shape is (B,T)
    // input[1] is act[probs], shape is (B,T,V)
    // input[2] is model targets, shape is (B,T)
    // output[0] is grad_act[logits], shape is (B,T,V)

    int B = inputs[1].shape[0];
    int T = inputs[1].shape[1];
    int V = inputs[1].shape[2];

    for (int b = 0; b < B; b++)
    {
        float *dloss_bt = reinterpret_cast<float *>(inputs[0].ptr) + b * T;
        int *ix = reinterpret_cast<int *>(inputs[2].ptr) + b * T;
        for (int t = 0; t < T; t++)
        {
            float *dlogits_bt = reinterpret_cast<float *>(outputs[0].ptr) + b * T * V + t * V;
            float *probs_bt = reinterpret_cast<float *>(inputs[1].ptr) + b * T * V + t * V;
            float dloss = dloss_bt[t];
            int target = ix[t];
            for (int v = 0; v < V; v++)
            {
                float indicator = v == target ? 1.0f : 0.0f;
                dlogits_bt[v] += (probs_bt[v] - indicator) * dloss;
            }
        }
    }
}