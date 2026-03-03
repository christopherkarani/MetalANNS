#include <metal_stdlib>
using namespace metal;

constant uint LOCKED_SLOT_F16 = 0xFFFFFFFEu;

kernel void compute_initial_distances_f16(
    device const half *vectors [[buffer(0)]],
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
            float va = float(vectors[base_a + d]);
            float vb = float(vectors[base_b + d]);
            dot += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }
        float denom = sqrt(norm_a) * sqrt(norm_b);
        result = (denom < 1e-10f) ? 1.0f : (1.0f - dot / denom);
    } else if (metric_type == 1u) {
        for (uint d = 0; d < dim; d++) {
            float diff = float(vectors[base_a + d]) - float(vectors[base_b + d]);
            result += diff * diff;
        }
    } else {
        float dot = 0.0f;
        for (uint d = 0; d < dim; d++) {
            dot += float(vectors[base_a + d]) * float(vectors[base_b + d]);
        }
        result = -dot;
    }

    distances[tid] = result;
}

inline float compute_metric_distance_f16(
    device const half *vectors,
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
            float va = float(vectors[base_a + d]);
            float vb = float(vectors[base_b + d]);
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
            float diff = float(vectors[base_a + d]) - float(vectors[base_b + d]);
            sum += diff * diff;
        }
        return sum;
    }

    float dot = 0.0f;
    for (uint d = 0; d < dim; d++) {
        dot += float(vectors[base_a + d]) * float(vectors[base_b + d]);
    }
    return -dot;
}

inline bool try_insert_neighbor_f16(
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

    uint worst_slot = UINT_MAX;
    uint worst_id = UINT_MAX;
    float worst_distance = -FLT_MAX;
    for (uint slot = 0; slot < degree; slot++) {
        uint slot_id = atomic_load_explicit(&adj_ids[base + slot], memory_order_relaxed);
        if (slot_id == LOCKED_SLOT_F16) {
            continue;
        }
        uint bits = atomic_load_explicit(&adj_dists_bits[base + slot], memory_order_relaxed);
        float distance = as_type<float>(bits);
        if (worst_slot == UINT_MAX || distance > worst_distance) {
            worst_distance = distance;
            worst_slot = slot;
            worst_id = slot_id;
        }
    }

    if (worst_slot == UINT_MAX) {
        return false;
    }
    if (candidate_distance >= worst_distance) {
        return false;
    }

    uint expected_id = worst_id;
    bool locked = atomic_compare_exchange_weak_explicit(
        &adj_ids[base + worst_slot],
        &expected_id,
        LOCKED_SLOT_F16,
        memory_order_relaxed,
        memory_order_relaxed
    );
    if (!locked) {
        return false;
    }

    uint observed_bits = atomic_load_explicit(&adj_dists_bits[base + worst_slot], memory_order_relaxed);
    float observed_distance = as_type<float>(observed_bits);
    if (candidate_distance >= observed_distance) {
        atomic_store_explicit(&adj_ids[base + worst_slot], expected_id, memory_order_relaxed);
        return false;
    }

    atomic_store_explicit(
        &adj_dists_bits[base + worst_slot],
        as_type<uint>(candidate_distance),
        memory_order_relaxed
    );
    atomic_store_explicit(&adj_ids[base + worst_slot], candidate, memory_order_relaxed);
    atomic_fetch_add_explicit(update_counter, 1u, memory_order_relaxed);
    return true;
}

inline float current_worst_distance_f16(
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
        if (slot_id == LOCKED_SLOT_F16) {
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

kernel void local_join_f16(
    device const half *vectors [[buffer(0)]],
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

        for (uint ri = 0; ri < actual_reverse; ri++) {
            uint b = rev[ri];
            if (b >= node_count || a == b) {
                continue;
            }

            float a_worst = current_worst_distance_f16(adj_ids, adj_dists_bits, a, degree);
            float b_worst = current_worst_distance_f16(adj_ids, adj_dists_bits, b, degree);
            float pair_dist = compute_metric_distance_f16(vectors, a, b, dim, metric_type);

            if (pair_dist < a_worst) {
                try_insert_neighbor_f16(
                    adj_ids,
                    adj_dists_bits,
                    a,
                    b,
                    node_count,
                    degree,
                    pair_dist,
                    update_counter
                );
            }

            if (pair_dist < b_worst) {
                try_insert_neighbor_f16(
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
