#include <metal_stdlib>
using namespace metal;

kernel void cosine_distance_f16(
    device const half *query [[buffer(0)]],
    device const half *corpus [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) {
        return;
    }

    float dot = 0.0f;
    float normQSq = 0.0f;
    float normVSq = 0.0f;

    uint base = tid * dim;
    for (uint d = 0; d < dim; d++) {
        float q = float(query[d]);
        float v = float(corpus[base + d]);
        dot += q * v;
        normQSq += q * q;
        normVSq += v * v;
    }

    float denom = sqrt(normQSq) * sqrt(normVSq);
    output[tid] = (denom < 1e-10f) ? 1.0f : (1.0f - (dot / denom));
}

kernel void l2_distance_f16(
    device const half *query [[buffer(0)]],
    device const half *corpus [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) {
        return;
    }

    float sumSq = 0.0f;
    uint base = tid * dim;
    for (uint d = 0; d < dim; d++) {
        float diff = float(query[d]) - float(corpus[base + d]);
        sumSq += diff * diff;
    }

    output[tid] = sumSq;
}

kernel void inner_product_distance_f16(
    device const half *query [[buffer(0)]],
    device const half *corpus [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) {
        return;
    }

    float dot = 0.0f;
    uint base = tid * dim;
    for (uint d = 0; d < dim; d++) {
        dot += float(query[d]) * float(corpus[base + d]);
    }

    output[tid] = -dot;
}
