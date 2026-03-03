#include <metal_stdlib>
using namespace metal;

constant uint LOCKED_SLOT = 0xFFFFFFFEu;

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

    uint base = tid * degree;
    if (node_count <= 1u) {
        for (uint slot = 0; slot < degree; slot++) {
            adjacency[base + slot] = uint_max;
        }
        return;
    }

    uint state = seed ^ (tid * 2654435761u);

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

inline float compute_metric_distance(
    device const float *vectors,
    uint a,
    uint b,
    uint dim,
    uint metric_type
) {
    uint base_a = a * dim;
    uint base_b = b * dim;

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
        return (denom < 1e-10f) ? 1.0f : (1.0f - dot / denom);
    }

    if (metric_type == 1u) {
        float sum = 0.0f;
        for (uint d = 0; d < dim; d++) {
            float diff = vectors[base_a + d] - vectors[base_b + d];
            sum += diff * diff;
        }
        return sum;
    }

    float dot = 0.0f;
    for (uint d = 0; d < dim; d++) {
        dot += vectors[base_a + d] * vectors[base_b + d];
    }
    return -dot;
}

inline bool try_insert_neighbor(
    device atomic_uint *adj_ids,
    device atomic_uint *adj_dists_bits,
    uint node,
    uint candidate,
    uint node_count,
    uint degree,
    float candidate_distance,
    device atomic_uint *update_counter
) {
    if (node >= node_count || candidate >= node_count || node == candidate) {
        return false;
    }

    uint base = node * degree;

    for (uint slot = 0; slot < degree; slot++) {
        uint current = atomic_load_explicit(&adj_ids[base + slot], memory_order_relaxed);
        if (current == candidate) {
            return false;
        }
    }

    uint worst_slot = uint_max;
    uint worst_id = uint_max;
    float worst_distance = -FLT_MAX;
    for (uint slot = 0; slot < degree; slot++) {
        uint slot_id = atomic_load_explicit(&adj_ids[base + slot], memory_order_relaxed);
        if (slot_id == LOCKED_SLOT) {
            continue;
        }
        uint bits = atomic_load_explicit(&adj_dists_bits[base + slot], memory_order_relaxed);
        float distance = as_type<float>(bits);
        if (worst_slot == uint_max || distance > worst_distance) {
            worst_distance = distance;
            worst_slot = slot;
            worst_id = slot_id;
        }
    }

    if (worst_slot == uint_max) {
        return false;
    }
    if (candidate_distance >= worst_distance) {
        return false;
    }

    uint expected_id = worst_id;
    bool locked = atomic_compare_exchange_weak_explicit(
        &adj_ids[base + worst_slot],
        &expected_id,
        LOCKED_SLOT,
        memory_order_relaxed,
        memory_order_relaxed
    );
    if (!locked) {
        return false;
    }

    uint observed_bits = atomic_load_explicit(&adj_dists_bits[base + worst_slot], memory_order_relaxed);
    float observed_distance = as_type<float>(observed_bits);
    if (candidate_distance >= observed_distance) {
        atomic_store_explicit(&adj_ids[base + worst_slot], expected_id, memory_order_release);
        return false;
    }

    atomic_store_explicit(
        &adj_dists_bits[base + worst_slot],
        as_type<uint>(candidate_distance),
        memory_order_relaxed
    );
    atomic_store_explicit(&adj_ids[base + worst_slot], candidate, memory_order_release);
    atomic_fetch_add_explicit(update_counter, 1u, memory_order_relaxed);
    return true;
}

inline float current_worst_distance(
    device atomic_uint *adj_ids,
    device atomic_uint *adj_dists_bits,
    uint node,
    uint degree
) {
    uint base = node * degree;
    float worst_distance = -FLT_MAX;
    bool found = false;
    for (uint slot = 0; slot < degree; slot++) {
        uint slot_id = atomic_load_explicit(&adj_ids[base + slot], memory_order_relaxed);
        if (slot_id == LOCKED_SLOT) {
            continue;
        }
        uint bits = atomic_load_explicit(&adj_dists_bits[base + slot], memory_order_relaxed);
        float distance = as_type<float>(bits);
        if (!found || distance > worst_distance) {
            worst_distance = distance;
            found = true;
        }
    }
    return found ? worst_distance : -FLT_MAX;
}

kernel void build_reverse_list(
    device const uint *adjacency [[buffer(0)]],
    device uint *reverse_list [[buffer(1)]],
    device atomic_uint *reverse_counts [[buffer(2)]],
    constant uint &node_count [[buffer(3)]],
    constant uint &degree [[buffer(4)]],
    constant uint &max_reverse [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = node_count * degree;
    if (tid >= total) {
        return;
    }

    uint source = tid / degree;
    uint target = adjacency[tid];
    if (target >= node_count) {
        return;
    }

    uint slot = atomic_fetch_add_explicit(&reverse_counts[target], 1u, memory_order_relaxed);
    if (slot < max_reverse) {
        reverse_list[target * max_reverse + slot] = source;
    }
}

kernel void local_join(
    device const float *vectors [[buffer(0)]],
    device atomic_uint *adj_ids [[buffer(1)]],
    device atomic_uint *adj_dists_bits [[buffer(2)]],
    device const uint *reverse_list [[buffer(3)]],
    device const uint *reverse_counts_r [[buffer(4)]],
    constant uint &node_count [[buffer(5)]],
    constant uint &degree [[buffer(6)]],
    constant uint &max_reverse [[buffer(7)]],
    constant uint &dim [[buffer(8)]],
    constant uint &metric_type [[buffer(9)]],
    device atomic_uint *update_counter [[buffer(10)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= node_count) {
        return;
    }

    uint fwd[64];
    uint fwd_count = 0u;
    uint fwd_base = tid * degree;
    for (uint slot = 0; slot < degree && fwd_count < 64u; slot++) {
        uint neighbor = atomic_load_explicit(&adj_ids[fwd_base + slot], memory_order_relaxed);
        if (neighbor < node_count && neighbor != tid) {
            fwd[fwd_count++] = neighbor;
        }
    }

    uint rev[128];
    uint reverse_count = reverse_counts_r[tid];
    uint actual_reverse = min(min(reverse_count, max_reverse), 128u);
    uint rev_base = tid * max_reverse;
    for (uint idx = 0; idx < actual_reverse; idx++) {
        rev[idx] = reverse_list[rev_base + idx];
    }

    for (uint fi = 0; fi < fwd_count; fi++) {
        uint a = fwd[fi];
        if (a >= node_count) {
            continue;
        }

        float a_worst = current_worst_distance(adj_ids, adj_dists_bits, a, degree);

        for (uint ri = 0; ri < actual_reverse; ri++) {
            uint b = rev[ri];
            if (b >= node_count || a == b) {
                continue;
            }

            float b_worst = current_worst_distance(adj_ids, adj_dists_bits, b, degree);
            float pair_dist = compute_metric_distance(vectors, a, b, dim, metric_type);

            if (pair_dist < a_worst) {
                bool inserted = try_insert_neighbor(
                    adj_ids,
                    adj_dists_bits,
                    a,
                    b,
                    node_count,
                    degree,
                    pair_dist,
                    update_counter
                );
                if (inserted) {
                    a_worst = current_worst_distance(adj_ids, adj_dists_bits, a, degree);
                }
            }

            if (pair_dist < b_worst) {
                try_insert_neighbor(
                    adj_ids,
                    adj_dists_bits,
                    b,
                    a,
                    node_count,
                    degree,
                    pair_dist,
                    update_counter
                );
            }
        }
    }
}
