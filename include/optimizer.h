#include "common.h"
#include "net.h"

class AdamwOptimizer
{
public:
    Net *net;
    float learning_rate = 1e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    float weight_decay = 0.0f;
    float *m_memory = nullptr;
    float *v_memory = nullptr;

    AdamwOptimizer(Net *net_ptr) : net(net_ptr){};
    AdamwOptimizer(Net *net_ptr, float lr, float beta1_, float beta2_, float epsilon_, float wd)
        : net(net_ptr), learning_rate(lr), beta1(beta1_), beta2(beta2_), epsilon(epsilon_), weight_decay(wd) {}

    void zero_grad();
    void update(int t);
};