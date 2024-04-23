#include "dataloader.h"

Dataloader::Dataloader(const string &filename, int batch_size_, int seq_len_) noexcept
    : batch_size(batch_size_), seq_len(seq_len_), file(), file_size(0), current_pos(0),
      batch(nullptr), inputs(nullptr), targets(nullptr), num_batches(0)
{
    // Load the data
    file.open(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "Error: could not open file " << filename << endl;
        exit(1);
    }
    file.seekg(0, ios::end);
    file_size = file.tellg();
    file.seekg(0, ios::beg);

    if (file_size < (batch_size * seq_len + 1) * sizeof(int))
    {
        cerr << "Error: file size is too small for the given batch size and sequence length" << endl;
        exit(1);
    }
    batch = new int[batch_size * seq_len + 1];
    inputs = batch;
    targets = batch + 1;
    num_batches = file_size / (batch_size * seq_len * sizeof(int));
}

void Dataloader::reset()
{
    current_pos = 0;
}

void Dataloader::next_batch()
{
    // 如果取下一个batch超出文件长度，则重置loader的current_position
    if (current_pos + (batch_size * seq_len + 1) * sizeof(int) > file_size)
    {
        reset();
    }
    file.seekg(current_pos, ios::beg);
    file.read(reinterpret_cast<char *>(batch), (batch_size * seq_len + 1) * sizeof(int));
    current_pos += batch_size * seq_len * sizeof(int);
}