#include "common.h"
#include "tensor.h"

class Ops
{
public:
    vector<TensorObj> inputs;
    vector<TensorObj> outputs;

    Ops(vector<TensorObj> inputs_, vector<TensorObj> outputs_) : inputs(inputs_), outputs(outputs_){};
    virtual void forward() = 0;
    virtual void backward() = 0;
};

class EncoderOp : public Ops
{
public:
    EncoderOp(vector<TensorObj> inputs, vector<TensorObj> outputs);
    void forward() override;
    void backward() override;
};

class LayerNormOp : public Ops
{
public:
    LayerNormOp(vector<TensorObj> inputs, vector<TensorObj> outputs);
    void forward() override;
    void backward() override;
};

class MatMulOp : public Ops
{
public:
    MatMulOp(vector<TensorObj> inputs, vector<TensorObj> outputs);
    void forward() override;
    void backward() override;
};

class AttentionOp : public Ops
{
public:
    AttentionOp(vector<TensorObj> inputs, vector<TensorObj> outputs);
    void forward() override;
    void backward() override;
};

class ResidualOp : public Ops
{
public:
    ResidualOp(vector<TensorObj> inputs, vector<TensorObj> outputs);
    void forward() override;
    void backward() override;
};

class GeluOp : public Ops
{
public:
    GeluOp(vector<TensorObj> inputs, vector<TensorObj> outputs);
    void forward() override;
    void backward() override;
};

class SoftmaxOp : public Ops
{
public:
    SoftmaxOp(vector<TensorObj> inputs, vector<TensorObj> outputs);
    void forward() override;
    void backward() override;
};

class CrossEntropyOp : public Ops
{
public:
    CrossEntropyOp(vector<TensorObj> inputs, vector<TensorObj> outputs);
    void forward() override;
    void backward() override;
};
