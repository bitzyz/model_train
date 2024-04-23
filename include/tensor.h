#pragma once
#include "common.h"

using std::vector;

using Shape = vector<int>;

// struct DataType
// {
//     enum : uint8_t
//     {
//         F32 = 1,
//         I32 = 2,
//     } type;
//     DataType() { type = F32; }
//     constexpr DataType(int i) noexcept : type(i) {}
// };

// template <int index>
// struct DT
// {
// };
// template <>
// struct DT<1>
// {
//     using t = float;
// };
// template <>
// struct DT<2>
// {
//     using t = int32_t;
// };

class TensorObj
{
public:
    Shape shape;
    // DataType datatype;
    void *ptr;

public:
    TensorObj() {}
    TensorObj(Shape shape_) : shape(shape_)
    {
        auto size = 1;
        for (auto s : shape)
        {
            size *= s;
        }
        ptr = (void *)malloc(sizeof(float) * size);
    };

    TensorObj(Shape shape_, void *ptr_) : shape(shape_), ptr(ptr_) {}

    size_t size()
    {
        auto size = 1;
        for (auto s : shape)
        {
            size *= s;
        }
        return size;
    }

    // ~TensorObj()
    // {
    //     delete ptr;
    // }
};
