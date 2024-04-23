#pragma once
#include "common.h"

class Dataloader
{
public:
    int batch_size;
    int seq_len;
    std::ifstream file;
    long file_size;
    long current_pos;
    int *batch;
    int *inputs;
    int *targets;

    int num_batches;

    Dataloader(const std::string &filename, int batch_size, int seq_len) noexcept;
    void reset();
    void next_batch();

    ~Dataloader()
    {
        delete batch;
        file.close();
    }
};