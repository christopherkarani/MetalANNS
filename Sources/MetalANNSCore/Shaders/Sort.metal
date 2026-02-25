#include <metal_stdlib>
using namespace metal;

#define MAX_DEGREE 64

kernel void bitonic_sort_neighbors(
    device uint *adjacency [[buffer(0)]],
    device float *distances [[buffer(1)]],
    constant uint &degree [[buffer(2)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (degree == 0u || degree > MAX_DEGREE) {
        return;
    }

    uint pair_count = degree >> 1;
    if (tid >= pair_count) {
        return;
    }

    uint node = tg_pos.x;
    uint base = node * degree;

    threadgroup float shared_dists[MAX_DEGREE];
    threadgroup uint shared_ids[MAX_DEGREE];

    uint first = tid;
    uint second = tid + pair_count;

    shared_dists[first] = distances[base + first];
    shared_ids[first] = adjacency[base + first];
    shared_dists[second] = distances[base + second];
    shared_ids[second] = adjacency[base + second];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2u; k <= degree; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            uint i = ((tid / j) * (j << 1u)) + (tid % j);
            uint ixj = i + j;

            if (ixj < degree) {
                bool ascending = (i & k) == 0u;
                float left = shared_dists[i];
                float right = shared_dists[ixj];
                bool shouldSwap = ascending ? (left > right) : (left < right);

                if (shouldSwap) {
                    shared_dists[i] = right;
                    shared_dists[ixj] = left;

                    uint leftID = shared_ids[i];
                    shared_ids[i] = shared_ids[ixj];
                    shared_ids[ixj] = leftID;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    distances[base + first] = shared_dists[first];
    adjacency[base + first] = shared_ids[first];
    distances[base + second] = shared_dists[second];
    adjacency[base + second] = shared_ids[second];
}
