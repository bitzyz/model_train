#include "net.h"
#include "ops.h"

Net::Net(const string &check_point_path) noexcept
{
    ifstream file(check_point_path, ios::binary);
    if (!file)
    {
        cerr << "Error: Could not open file " << check_point_path << endl;
        exit(1);
    }
    const int numIntsToRead = 256;
    vector<int> modelHeaders(numIntsToRead);
    file.read(reinterpret_cast<char *>(modelHeaders.data()), numIntsToRead * sizeof(int));
    if (modelHeaders[0] != 20240326)
    {
        cerr << "Bad magic model file " << endl;
        exit(1);
    }
    if (modelHeaders[1] != 1)
    {
        cerr << "Bad version in model file" << endl;
        exit(1);
    }
    // init net config
    int max_sequence_length, vocab_size, num_layers, num_heads, embedding_size;
    config.max_sequence_length = max_sequence_length = modelHeaders[2];
    config.vocab_size = vocab_size = modelHeaders[3];
    config.num_layers = num_layers = modelHeaders[4];
    config.num_heads = num_heads = modelHeaders[5];
    config.embedding_size = embedding_size = modelHeaders[6];

    // print model config
    cout << "max_seq_len : " << config.max_sequence_length << endl;
    cout << "vocab_size : " << config.vocab_size << endl;
    cout << "num_layers : " << config.num_layers << endl;
    cout << "num_heads : " << config.num_heads << endl;
    cout << "embedding_size : " << config.embedding_size << endl;

    // init net params
    params.addParam("wte", TensorObj({vocab_size, embedding_size}));
    params.addParam("wpe", TensorObj({max_sequence_length, embedding_size}));
    params.addParam("ln1w", TensorObj({num_layers, embedding_size}));
    params.addParam("ln1b", TensorObj({num_layers, embedding_size}));
    params.addParam("qkvw", TensorObj({num_layers, 3 * embedding_size, embedding_size}));
    params.addParam("qkvb", TensorObj({num_layers, 3 * embedding_size}));
    params.addParam("attprojw", TensorObj({num_layers, embedding_size, embedding_size}));
    params.addParam("attprojb", TensorObj({num_layers, embedding_size}));
    params.addParam("ln2w", TensorObj({num_layers, embedding_size}));
    params.addParam("ln2b", TensorObj({num_layers, embedding_size}));
    params.addParam("fcw", TensorObj({num_layers, 4 * embedding_size, embedding_size}));
    params.addParam("fcb", TensorObj({num_layers, 4 * embedding_size}));
    params.addParam("fcprojw", TensorObj({num_layers, embedding_size, 4 * embedding_size}));
    params.addParam("fcprojb", TensorObj({num_layers, embedding_size}));
    params.addParam("lnfw", TensorObj({embedding_size}));
    params.addParam("lnfb", TensorObj({embedding_size}));

    // this->read_params_from_file(file);

    size_t num_params_ = 0;
    for (auto &param : params.params_)
    {
        auto size = param.second.size();
        auto totalBytes = size * sizeof(float);
        num_params_ += size;
        file.read(reinterpret_cast<char *>(param.second.ptr), totalBytes);
        if (file.gcount() != totalBytes)
        {
            cerr << "Read model params failed " << endl;
            exit(1);
        }
    }
    num_params = num_params_;
    cout << "num_params : " << num_params << endl;
    file.close();
}

void Net::forward(int *inputs, int *targets, int B, int T)
{
    int V = config.vocab_size;
    int L = config.num_layers;
    int H = config.num_heads;
    int C = config.embedding_size;

    // init acts
    if (num_acts == 0)
    {
        acts.addAct("encoded", TensorObj({B, T, C}));         // 输入x(B,T)->embedding->x_state(B,T,C)
        acts.addAct("ln1", TensorObj({L, B, T, C}));          // x_state(B,T,C)->layernorm1->x1_state(B,T,C)
        acts.addAct("ln1_mean", TensorObj({L, B, T}));        // layernorm1中间用到的mean
        acts.addAct("ln1_rstd", TensorObj({L, B, T}));        // layernorm1中间用到的var
        acts.addAct("qkv", TensorObj({L, B, T, 3 * C}));      // x1_state(B,T,C)->attn->qkv_state(B,T,3C)
        acts.addAct("preatt", TensorObj({L, B, H, T, T}));    // 经过mask后的q @ k.T (B,NH,T,NS) @ (B, NH, NS, T)
        acts.addAct("att", TensorObj({L, B, H, T, T}));       // attn_score = softmax(q @ k.T)
        acts.addAct("atty", TensorObj({L, B, T, C}));         // o = attn_score @ v的输出
        acts.addAct("attproj", TensorObj({L, B, T, C}));      // 输出y = o * wo + bo
        acts.addAct("residual2", TensorObj({L, B, T, C}));    // 经过layernorm1和attn层之后，经过残差连接之后的输出
        acts.addAct("ln2", TensorObj({L, B, T, C}));          // 经过layernorm2层的输出
        acts.addAct("ln2_mean", TensorObj({L, B, T}));        // layernorm2中间用到的mean
        acts.addAct("ln2_rstd", TensorObj({L, B, T}));        // layernorm2中间用到的var
        acts.addAct("fch", TensorObj({L, B, T, 4 * C}));      // mlp经过第一个linear层的输出
        acts.addAct("fch_gelu", TensorObj({L, B, T, 4 * C})); // mlp经过gelu层的输出
        acts.addAct("fcproj", TensorObj({L, B, T, C}));       // mlp经过第二个linear层的输出
        acts.addAct("residual3", TensorObj({L, B, T, C}));    // 经过layernorm2和mlp层之后，经过残差连接之后的输出
        acts.addAct("lnf", TensorObj({B, T, C}));             // 经过最后一层layernorm层的输出
        acts.addAct("lnf_mean", TensorObj({B, T}));           // layernorm中间用到的mean
        acts.addAct("lnf_rstd", TensorObj({B, T}));           // layernorm中间用到的var
        acts.addAct("logits", TensorObj({B, T, V}));          // 最后网络结果输出对应词表的值
        acts.addAct("probs", TensorObj({B, T, V}));           // 经过softmax之后的输出概率值
        acts.addAct("losses", TensorObj({B, T}));
        size_t num_acts_ = 0;
        for (auto &act : acts.acts_)
        {
            auto size = act.second.size();
            num_acts_ += size;
        }
        num_acts = num_acts_;
        cout << "num_acts : " << num_acts << endl;
    }
    auto input = TensorObj({B, T}, reinterpret_cast<void *>(inputs));

    // forward
    {
        EncoderOp({input, params["wte"], params["wpe"]}, {acts["encoded"]}).forward();
        for (int l = 0; l < L; l++)
        {
            auto residual = l == 0 ? acts["encoded"] : TensorObj({B, T, C}, acts["residual3"].ptr + (l - 1) * B * T * C * sizeof(float));
            auto ln1 = TensorObj({B, T, C}, acts["ln1"].ptr + l * B * T * C * sizeof(float));
            LayerNormOp({residual, TensorObj({C}, params["ln1w"].ptr + l * C * sizeof(float)), TensorObj({C}, params["ln1b"].ptr + l * C * sizeof(float))},
                        {ln1, TensorObj({B, T}, acts["ln1_mean"].ptr + l * B * T * sizeof(float)), TensorObj({B, T}, acts["ln1_rstd"].ptr + l * B * T * sizeof(float))})
                .forward();
            auto qkv = TensorObj({B, T, 3 * C}, acts["qkv"].ptr + l * B * T * 3 * C * sizeof(float));
            MatMulOp({ln1, TensorObj({3 * C, C}, params["qkvw"].ptr + l * 3 * C * C * sizeof(float)), TensorObj({3 * C}, params["qkvb"].ptr + l * 3 * C * sizeof(float))}, {qkv}).forward();
            auto preatt = TensorObj({B, H, T, T}, acts["preatt"].ptr + l * B * H * T * T * sizeof(float));
            auto att = TensorObj({B, H, T, T}, acts["att"].ptr + l * B * H * T * T * sizeof(float));
            auto atty = TensorObj({B, T, C}, acts["atty"].ptr + l * B * T * C * sizeof(float));
            AttentionOp({qkv}, {atty, preatt, att}).forward();
            auto attproj = TensorObj({B, T, C}, acts["attproj"].ptr + l * B * T * C * sizeof(float));
            MatMulOp({atty, TensorObj({C, C}, params["attprojw"].ptr + l * C * C * sizeof(float)), TensorObj({C}, params["attprojb"].ptr + l * C * sizeof(float))}, {attproj}).forward();
            auto residual2 = TensorObj({B, T, C}, acts["residual2"].ptr + l * B * T * C * sizeof(float));
            ResidualOp({attproj, residual}, {residual2}).forward();
            auto ln2 = TensorObj({B, T, C}, acts["ln2"].ptr + l * B * T * C * sizeof(float));
            LayerNormOp({residual2, TensorObj({C}, params["ln2w"].ptr + l * C * sizeof(float)), TensorObj({C}, params["ln2b"].ptr + l * C * sizeof(float))},
                        {ln2, TensorObj({B, T}, acts["ln2_mean"].ptr + l * B * T * sizeof(float)), TensorObj({B, T}, acts["ln2_rstd"].ptr + l * B * T * sizeof(float))})
                .forward();
            auto fch = TensorObj({B, T, 4 * C}, acts["fch"].ptr + l * B * T * 4 * C * sizeof(float));
            MatMulOp({ln2, TensorObj({4 * C, C}, params["fcw"].ptr + l * 4 * C * C * sizeof(float)), TensorObj({4 * C}, params["fcb"].ptr + l * 4 * C * sizeof(float))}, {fch}).forward();
            auto gelufch = TensorObj({B, T, 4 * C}, acts["fch_gelu"].ptr + l * B * T * 4 * C * sizeof(float));
            GeluOp({fch}, {gelufch}).forward();
            auto fcproj = TensorObj({B, T, C}, acts["fcproj"].ptr + l * B * T * C * sizeof(float));
            MatMulOp({gelufch, TensorObj({C, 4 * C}, params["fcprojw"].ptr + l * C * 4 * C * sizeof(float)), TensorObj({C}, params["fcprojb"].ptr + l * C * sizeof(float))}, {fcproj}).forward();
            auto residual3 = TensorObj({B, T, C}, acts["residual3"].ptr + l * B * T * C * sizeof(float));
            ResidualOp({fcproj, residual2}, {residual3}).forward();
        }
        auto residual = TensorObj({B, T, C}, acts["residual3"].ptr + (L - 1) * B * T * C * sizeof(float));
        LayerNormOp({residual, params["lnfw"], params["lnfb"]}, {acts["lnf"], acts["lnf_mean"], acts["lnf_rstd"]}).forward();
        MatMulOp({acts["lnf"], params["wte"]}, {acts["logits"]}).forward();
        SoftmaxOp({acts["logits"]}, {acts["probs"]}).forward();

        if (targets)
        {
            auto target = TensorObj({B, T}, reinterpret_cast<void *>(targets));
            CrossEntropyOp({acts["probs"], target}, {acts["losses"]}).forward();
            float sum_loss_ = 0.0f;
            for (int i = 0; i < B * T; i++)
            {
                sum_loss_ += reinterpret_cast<float *>(acts["losses"].ptr)[i];
            }
            mean_loss = sum_loss_ / (B * T);
        }
    }
}

void Net::backward(int *inputs, int *targets, int B, int T)
{
    if (mean_loss == -1.0f)
    {
        cerr << "Error: must forward with targets before backward" << endl;
        exit(1);
    }
    if (param_grads.params_.size() == 0 || act_grads.acts_.size() == 0)
    {
        cerr << "Error: must init param_grads before backward" << endl;
        exit(1);
    }

    int V = config.vocab_size;
    int L = config.num_layers;
    int H = config.num_heads;
    int C = config.embedding_size;

    // we kick off the chain by filling in dlosses with 1.0f/(B*T), to get the mean loss
    float dloss_mean = 1.0f / (B * T);
    float *dlosses = reinterpret_cast<float *>(act_grads["losses"].ptr);
    for (int i = 0; i < B * T; i++)
    {
        dlosses[i] = dloss_mean;
    }

    auto target = TensorObj({B, T}, reinterpret_cast<void *>(targets));
    auto input = TensorObj({B, T}, reinterpret_cast<void *>(inputs));
    // backward
    {
        CrossEntropyOp({act_grads["losses"], acts["probs"], target}, {act_grads["logits"]}).backward();
        MatMulOp({act_grads["logits"], acts["lnf"], params["wte"]}, {act_grads["lnf"], param_grads["wte"]}).backward();
        auto residual = TensorObj({B, T, C}, acts["residual3"].ptr + (L - 1) * B * T * C * sizeof(float));
        auto dresidual = TensorObj({B, T, C}, act_grads["residual3"].ptr + (L - 1) * B * T * C * sizeof(float));
        LayerNormOp({act_grads["lnf"], residual, params["lnfw"], acts["lnf_mean"], acts["lnf_rstd"]}, {dresidual, param_grads["lnfw"], param_grads["lnfb"]}).backward();
        for (int l = L - 1; l >= 0; l--)
        {
            residual = l == 0 ? acts["encoded"] : TensorObj({B, T, C}, acts["residual3"].ptr + (l - 1) * B * T * C * sizeof(float));
            dresidual = l == 0 ? act_grads["encoded"] : TensorObj({B, T, C}, act_grads["residual3"].ptr + (l - 1) * B * T * C * sizeof(float));

            auto dresidual3 = TensorObj({B, T, C}, act_grads["residual3"].ptr + l * B * T * C * sizeof(float));
            auto dfcproj = TensorObj({B, T, C}, act_grads["fcproj"].ptr + l * B * T * C * sizeof(float));
            auto dresidual2 = TensorObj({B, T, C}, act_grads["residual2"].ptr + l * B * T * C * sizeof(float));
            ResidualOp({dresidual3}, {dresidual2, dfcproj}).backward();
            auto dfch_gelu = TensorObj({B, T, 4 * C}, act_grads["fch_gelu"].ptr + l * B * T * 4 * C * sizeof(float));
            auto dfcprojw = TensorObj({C, 4 * C}, param_grads["fcprojw"].ptr + l * C * 4 * C * sizeof(float));
            auto dfcprojb = TensorObj({C}, param_grads["fcprojb"].ptr + l * C * sizeof(float));
            MatMulOp({dfcproj, TensorObj({B, T, 4 * C}, acts["fch_gelu"].ptr + l * B * T * 4 * C * sizeof(float)), TensorObj({C, 4 * C}, params["fcprojw"].ptr + l * C * 4 * C * sizeof(float))}, {dfch_gelu, dfcprojw, dfcprojb}).backward();
            auto dfch = TensorObj({B, T, 4 * C}, act_grads["fch"].ptr + l * B * T * 4 * C * sizeof(float));
            GeluOp({dfch_gelu, TensorObj({B, T, 4 * C}, acts["fch"].ptr + l * B * T * 4 * C * sizeof(float))}, {dfch}).backward();
            auto dln2 = TensorObj({B, T, C}, act_grads["ln2"].ptr + l * B * T * C * sizeof(float));
            auto dfcw = TensorObj({4 * C, C}, param_grads["fcw"].ptr + l * 4 * C * C * sizeof(float));
            auto dfcb = TensorObj({4 * C}, param_grads["fcb"].ptr + l * 4 * C * sizeof(float));
            MatMulOp({dfch, TensorObj({B, T, C}, acts["ln2"].ptr + l * B * T * C * sizeof(float)), TensorObj({4 * C, C}, params["fcw"].ptr + l * 4 * C * C * sizeof(float))}, {dln2, dfcw, dfcb}).backward();
            auto dln2w = TensorObj({C}, param_grads["ln2w"].ptr + l * C * sizeof(float));
            auto dln2b = TensorObj({C}, param_grads["ln2b"].ptr + l * C * sizeof(float));
            LayerNormOp({dln2,
                         TensorObj({B, T, C}, acts["residual2"].ptr + l * B * T * C * sizeof(float)),
                         TensorObj({C}, params["ln2w"].ptr + l * C * sizeof(float)),
                         TensorObj({B, T}, acts["ln2_mean"].ptr + l * B * T * sizeof(float)),
                         TensorObj({B, T}, acts["ln2_rstd"].ptr + l * B * T * sizeof(float))},
                        {dresidual2,
                         dln2w,
                         dln2b})
                .backward();
            auto dattproj = TensorObj({B, T, C}, act_grads["attproj"].ptr + l * B * T * C * sizeof(float));
            ResidualOp({dresidual2}, {dresidual, dattproj}).backward();
            auto dattprojw = TensorObj({C, C}, param_grads["attprojw"].ptr + l * C * C * sizeof(float));
            auto dattprojb = TensorObj({C}, param_grads["attprojb"].ptr + l * C * sizeof(float));
            auto datty = TensorObj({B, T, C}, act_grads["atty"].ptr + l * B * T * C * sizeof(float));
            MatMulOp({dattproj, TensorObj({B, T, C}, acts["atty"].ptr + l * B * T * C * sizeof(float)), TensorObj({C, C}, params["attprojw"].ptr + l * C * C * sizeof(float))}, {datty, dattprojw, dattprojb})
                .backward();
            auto dqkv = TensorObj({B, T, 3 * C}, act_grads["qkv"].ptr + l * B * T * 3 * C * sizeof(float));
            auto dpreatt = TensorObj({B, H, T, T}, act_grads["preatt"].ptr + l * B * H * T * T * sizeof(float));
            auto datt = TensorObj({B, H, T, T}, act_grads["att"].ptr + l * B * H * T * T * sizeof(float));
            AttentionOp({datty, TensorObj({B, T, 3 * C}, acts["qkv"].ptr + l * B * T * 3 * C * sizeof(float)), TensorObj({B, H, T, T}, acts["att"].ptr + l * B * H * T * T * sizeof(float))}, {dqkv, dpreatt, datt}).backward();
            auto dln1 = TensorObj({B, T, C}, act_grads["ln1"].ptr + l * B * T * C * sizeof(float));
            auto dqkvw = TensorObj({3 * C, C}, param_grads["qkvw"].ptr + l * 3 * C * C * sizeof(float));
            auto dqkvb = TensorObj({3 * C}, param_grads["qkvb"].ptr + l * 3 * C * sizeof(float));
            MatMulOp({dqkv, TensorObj({B, T, C}, acts["ln1"].ptr + l * B * T * C * sizeof(float)), TensorObj({3 * C, C}, params["qkvw"].ptr + l * 3 * C * C * sizeof(float))}, {dln1, dqkvw, dqkvb}).backward();
            auto dln1w = TensorObj({C}, param_grads["ln1w"].ptr + l * C * sizeof(float));
            auto dln1b = TensorObj({C}, param_grads["ln1b"].ptr + l * C * sizeof(float));
            LayerNormOp({dln1,
                         residual,
                         TensorObj({C}, params["ln1w"].ptr + l * C * sizeof(float)),
                         TensorObj({B, T}, acts["ln1_mean"].ptr + l * B * T * sizeof(float)),
                         TensorObj({B, T}, acts["ln1_rstd"].ptr + l * B * T * sizeof(float))},
                        {dresidual, dln1w, dln1b})
                .backward();
        }
        EncoderOp({act_grads["encoded"], input}, {param_grads["wte"], param_grads["wpe"]}).backward();
    }
}