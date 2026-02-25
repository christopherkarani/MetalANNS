#include <metal_stdlib>
using namespace metal;

kernel void cosine_distance(
    device const float *query [[buffer(0)]],
    device const float *corpus [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) {
        return;
    }

    float dot = 0.0f;
    float nq = 0.0f;
    float nv = 0.0f;
    uint base = tid * dim;
    for (uint i = 0; i < dim; i++) {
        float q = query[i];
        float v = corpus[base + i];
        dot += q * v;
        nq += q * q;
        nv += v * v;
    }
    float denom = sqrt(nq) * sqrt(nv);
    output[tid] = (denom < 1e-10f) ? 1.0f : (1.0f - (dot / denom));
}
