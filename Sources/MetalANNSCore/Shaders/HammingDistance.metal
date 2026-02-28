#include <metal_stdlib>
using namespace metal;

kernel void hamming_distance(
    device const uchar *query [[buffer(0)]],
    device const uchar *corpus [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &bytesPerVector [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) {
        return;
    }

    uint dist = 0;
    uint base = tid * bytesPerVector;
    for (uint index = 0; index < bytesPerVector; index++) {
        dist += popcount(query[index] ^ corpus[base + index]);
    }

    output[tid] = float(dist);
}
