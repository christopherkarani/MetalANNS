#include <metal_stdlib>
using namespace metal;

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

    uint worst_slot = 0u;
    uint worst_bits = atomic_load_explicit(&adj_dists_bits[base], memory_order_relaxed);
    float worst_distance = as_type<float>(worst_bits);
    for (uint slot = 1; slot < degree; slot++) {
        uint bits = atomic_load_explicit(&adj_dists_bits[base + slot], memory_order_relaxed);
        float distance = as_type<float>(bits);
        if (distance > worst_distance) {
            worst_bits = bits;
            worst_distance = distance;
            worst_slot = slot;
        }
    }

    if (candidate_distance >= worst_distance) {
        return false;
    }

    uint candidate_dist_bits = as_type<uint>(candidate_distance);
    uint expected = worst_bits;
    bool exchanged = atomic_compare_exchange_weak_explicit(
        &adj_dists_bits[base + worst_slot],
        &expected,
        candidate_dist_bits,
        memory_order_relaxed,
        memory_order_relaxed
    );

    if (exchanged) {
        atomic_store_explicit(&adj_ids[base + worst_slot], candidate, memory_order_relaxed);
        atomic_fetch_add_explicit(update_counter, 1u, memory_order_relaxed);
        return true;
    }

    return false;
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

            float pair_dist = compute_metric_distance_f16(vectors, a, b, dim, metric_type);

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
