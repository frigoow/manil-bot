#include "nn_policy.h"

#include "encoding.h"

#include <torch/script.h>
#include <torch/cuda.h>

#include <cmath>
#include <stdexcept>

struct NNPolicyValue::Impl {
    torch::jit::Module module;
    torch::Device device{torch::kCPU};
    bool half = false;
};

NNPolicyValue::NNPolicyValue(const std::string& path, const std::string& device)
    : m_impl(std::make_unique<Impl>())
{
    m_impl->module = torch::jit::load(path);
    m_impl->module.eval();
    if (device == "cuda") {
        if (!torch::cuda::is_available())
            throw std::runtime_error("NNPolicyValue: cuda requested but unavailable");
        m_impl->device = torch::Device(torch::kCUDA);
        m_impl->half = true;
        m_impl->module.to(torch::kHalf);
    }
    m_impl->module.to(m_impl->device);
}

NNPolicyValue::~NNPolicyValue() = default;

void NNPolicyValue::operator()(const float* obs, int n, float* probs,
                               float* values) {
    torch::NoGradGuard guard;

    auto input = torch::from_blob(
        const_cast<float*>(obs),
        {static_cast<long>(n), static_cast<long>(encoding::OBS_DIM)},
        torch::TensorOptions().dtype(torch::kFloat32));
    input = input.to(m_impl->device);
    if (m_impl->half)
        input = input.to(torch::kHalf);

    auto output = m_impl->module.forward({input}).toTuple();
    auto logits = output->elements()[0].toTensor()
                      .to(torch::kFloat32).cpu().contiguous();
    auto vals = output->elements()[1].toTensor()
                    .to(torch::kFloat32).cpu().contiguous();

    const float* logits_ptr = logits.data_ptr<float>();
    const float* vals_ptr = vals.data_ptr<float>();

    for (int i = 0; i < n; ++i) {
        const float* o = obs + static_cast<size_t>(i) * encoding::OBS_DIM;
        const float* l = logits_ptr + static_cast<size_t>(i) * 32;

        float max_logit = -1e30f;
        for (int c = 0; c < 32; ++c)
            if (o[encoding::LEGAL_MASK_OFF + c] >= 0.5f && l[c] > max_logit)
                max_logit = l[c];

        float sum = 0.0f;
        for (int c = 0; c < 32; ++c) {
            const bool legal = o[encoding::LEGAL_MASK_OFF + c] >= 0.5f;
            const float e = legal ? std::exp(l[c] - max_logit) : 0.0f;
            probs[static_cast<size_t>(i) * 32 + c] = e;
            sum += e;
        }
        if (sum <= 0.0f) {
            // Defensive: no legal mask bits — uniform over all 32 (engine
            // states always have >= 1 legal move, so this is unreachable).
            for (int c = 0; c < 32; ++c)
                probs[static_cast<size_t>(i) * 32 + c] = 1.0f / 32.0f;
        } else {
            for (int c = 0; c < 32; ++c)
                probs[static_cast<size_t>(i) * 32 + c] /= sum;
        }
        values[i] = vals_ptr[i];
    }
}
