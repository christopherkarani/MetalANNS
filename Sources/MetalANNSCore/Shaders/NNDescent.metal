#include <metal_stdlib>
using namespace metal;

inline uint lcg_next(uint state) {
    return state * 1664525u + 1013904223u;
}

kernel void random_init(
    device uint *adjacency [[buffer(0)]],
    constant uint &node_count [[buffer(1)]],
    constant uint &degree [[buffer(2)]],
    constant uint &seed [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= node_count) {
        return;
    }

    uint state = seed ^ (tid * 2654435761u);
    uint base = tid * degree;

    for (uint slot = 0; slot < degree; slot++) {
        bool valid = false;
        uint neighbor = tid;

        for (uint attempt = 0; attempt < 100 && !valid; attempt++) {
            state = lcg_next(state);
            neighbor = state % node_count;
            if (neighbor == tid) {
                continue;
            }

            valid = true;
            for (uint prev = 0; prev < slot; prev++) {
                if (adjacency[base + prev] == neighbor) {
                    valid = false;
                    break;
                }
            }
        }

        if (!valid) {
            uint fallback = (tid + slot + 1u) % node_count;
            while (fallback == tid) {
                fallback = (fallback + 1u) % node_count;
            }

            bool duplicate = true;
            for (uint scans = 0; scans < node_count && duplicate; scans++) {
                duplicate = false;
                for (uint prev = 0; prev < slot; prev++) {
                    if (adjacency[base + prev] == fallback) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate) {
                    fallback = (fallback + 1u) % node_count;
                    if (fallback == tid) {
                        fallback = (fallback + 1u) % node_count;
                    }
                }
            }
            neighbor = fallback;
        }

        adjacency[base + slot] = neighbor;
    }
}

kernel void compute_initial_distances(
    device const float *vectors [[buffer(0)]],
    device const uint *adjacency [[buffer(1)]],
    device float *distances [[buffer(2)]],
    constant uint &node_count [[buffer(3)]],
    constant uint &degree [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant uint &metric_type [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = node_count * degree;
    if (tid >= total) {
        return;
    }

    uint node = tid / degree;
    uint neighbor = adjacency[tid];
    if (neighbor >= node_count) {
        distances[tid] = FLT_MAX;
        return;
    }

    uint base_a = node * dim;
    uint base_b = neighbor * dim;

    float result = 0.0f;

    if (metric_type == 0u) {
        float dot = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        for (uint d = 0; d < dim; d++) {
            float va = vectors[base_a + d];
            float vb = vectors[base_b + d];
            dot += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }

        float denom = sqrt(norm_a) * sqrt(norm_b);
        result = (denom < 1e-10f) ? 1.0f : (1.0f - dot / denom);
    } else if (metric_type == 1u) {
        for (uint d = 0; d < dim; d++) {
            float diff = vectors[base_a + d] - vectors[base_b + d];
            result += diff * diff;
        }
    } else {
        float dot = 0.0f;
        for (uint d = 0; d < dim; d++) {
            dot += vectors[base_a + d] * vectors[base_b + d];
        }
        result = -dot;
    }

    distances[tid] = result;
}
