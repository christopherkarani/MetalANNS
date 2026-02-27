#include <metal_stdlib>
using namespace metal;

kernel void pq_compute_distance_table(
    device const float *query [[buffer(0)]],
    device const float *codebooks [[buffer(1)]],
    device float *distTable [[buffer(2)]],
    device const uint &M [[buffer(3)]],
    device const uint &Ks [[buffer(4)]],
    device const uint &subspaceDim [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint subspace = gid.x;
    uint centroid = gid.y;
    if (subspace >= M || centroid >= Ks) {
        return;
    }

    device const float *querySubspace = query + (subspace * subspaceDim);
    uint codebookOffset = (subspace * Ks * subspaceDim) + (centroid * subspaceDim);
    device const float *centroidPtr = codebooks + codebookOffset;

    float distance = 0.0f;
    for (uint d = 0; d < subspaceDim; d++) {
        float delta = querySubspace[d] - centroidPtr[d];
        distance += delta * delta;
    }

    distTable[subspace * Ks + centroid] = distance;
}

kernel void pq_adc_scan(
    device const uchar *codes [[buffer(0)]],
    device const float *distTable [[buffer(1)]],
    device float *distances [[buffer(2)]],
    device const uint &M [[buffer(3)]],
    device const uint &Ks [[buffer(4)]],
    device const uint &vectorCount [[buffer(5)]],
    threadgroup float *tgDistTable [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= vectorCount) {
        return;
    }

    uint totalTableValues = M * Ks;
    for (uint i = tid; i < totalTableValues; i += 32) {
        tgDistTable[i] = distTable[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float totalDistance = 0.0f;
    uint codeBase = gid * M;
    for (uint m = 0; m < M; m++) {
        uchar code = codes[codeBase + m];
        totalDistance += tgDistTable[(m * Ks) + code];
    }

    distances[gid] = totalDistance;
}
