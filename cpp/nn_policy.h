#ifndef MANILLE_NN_POLICY_H
#define MANILLE_NN_POLICY_H

#include <memory>
#include <string>

// In-process NN policy-value inference for BatchedISMCTS (strength_program.md
// C1): a traced ManilleNet (learn/export_ts.py) loaded via libtorch, so the
// search never crosses into Python per wave. Legal-masking (from the obs
// LEGAL_MASK_OFF block) and softmax happen here; values come from the net in
// [-1, 1]. FP16 on CUDA, FP32 on CPU.
class NNPolicyValue {
public:
    explicit NNPolicyValue(const std::string& path,
                           const std::string& device = "cuda");
    ~NNPolicyValue();

    // obs: n * OBS_DIM floats (C++ encoder layout). Fills probs (n*32,
    // legal-masked softmax) and values (n).
    void operator()(const float* obs, int n, float* probs, float* values);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;  // pimpl: torch headers stay in the .cpp
};

#endif // MANILLE_NN_POLICY_H
