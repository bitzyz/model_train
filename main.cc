#include "tensor.h"
#include "config.h"
#include "net.h"
#include "dataloader.h"
#include "utils.h"
#include "optimizer.h"
#include <chrono>

#define GPT2_EOT 50256
#define GEN_MAX_LENGTH 64

int main()
{
    string model_file_path = "/home/zhangyunze/workspace/llm-train/test_simplify/llama_train/gpt2_124M.bin";
    string train_file_path = "/home/zhangyunze/workspace/llm-train/llm.c/data/tiny_shakespeare_train.bin";
    string val_file_path = "/home/zhangyunze/workspace/llm-train/llm.c/data/tiny_shakespeare_val.bin";
    Net net(model_file_path);
    int B = 4;
    int T = 64;
    Dataloader train_loader(train_file_path, B, T);
    cout << "train dataset num_batches : " << train_loader.num_batches << endl;
    Dataloader val_loader(val_file_path, B, T);
    cout << "val dataset num_batches : " << val_loader.num_batches << endl;
    int val_num_batches = 10;

    int gen_tokens[GEN_MAX_LENGTH];
    unsigned long long rng_state = 1337;
    auto optimizer = AdamwOptimizer(&net);

    for (int step = 0; step <= 40; step++)
    {
        if (step % 10 == 0)
        {
            float val_loss = 0.0f;
            val_loader.reset();
            for (int i = 0; i < val_num_batches; ++i)
            {
                val_loader.next_batch();
                // auto start1 = chrono::high_resolution_clock::now();
                net.forward(val_loader.inputs, val_loader.targets, B, T);
                // auto end1 = chrono::high_resolution_clock::now();
                val_loss += net.mean_loss;
                // auto duration = std::chrono::duration_cast<std::chrono::seconds>(end1 - start1);
                // cout << "current i is " << i << " val loss is " << net.mean_loss << " cost time is " << duration.count() << "s" << endl;
            }
            val_loss /= val_num_batches;
            cout << "val loss is " << val_loss << endl;
        }

        // once in a while do model inference to print generated text
        if (step > 0 && step % 20 == 0)
        {
            gen_tokens[0] = GPT2_EOT;
            for (int i = 1; i < GEN_MAX_LENGTH; i++)
            {
                net.forward(gen_tokens, nullptr, 1, i);
                float *probs = reinterpret_cast<float *>(net.acts["probs"].ptr) + (i - 1) * net.config.vocab_size;
                float coin = random_f32(&rng_state);
                int next_token = sample_mult(probs, net.config.vocab_size, coin);
                gen_tokens[i] = next_token;
            }
            cout << "generated text: " << endl;
            for (int i = 1; i < GEN_MAX_LENGTH; i++)
            {
                if (gen_tokens[i] == GPT2_EOT)
                    break;
                cout << gen_tokens[i] << " ";
            }
            cout << endl;
        }

        // training
        auto start = chrono::high_resolution_clock::now();

        train_loader.next_batch();
        auto mid1 = chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(mid1 - start);
        cout << "load data cost time is " << duration1.count() << "ms" << endl;

        net.forward(train_loader.inputs, train_loader.targets, B, T);
        auto mid2 = chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(mid2 - mid1);
        cout << "forward cost time is " << duration2.count() << "ms" << endl;

        optimizer.zero_grad();
        auto mid3 = chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(mid3 - mid2);
        cout << "zero_grad cost time is " << duration3.count() << "ms" << endl;

        net.backward(train_loader.inputs, train_loader.targets, B, T);
        auto mid4 = chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(mid4 - mid3);
        cout << "backward cost time is " << duration4.count() << "ms" << endl;

        optimizer.update(step + 1);
        auto end = chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - mid4);
        cout << "current step is " << step << " train loss is " << net.mean_loss << " cost time is " << duration.count() << "ms" << endl;
        exit(1);
    }

    return 0;
}