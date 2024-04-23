#pragma once

class ModelConfig
{
public:
    int max_sequence_length;
    int vocab_size;
    int num_layers;
    int num_heads;
    int embedding_size;
};

class TrainConfig
{
public:
    int batch_size;
    int seq_len;
    float loss;
};
