#include <metal_stdlib>
using namespace metal;

constant uint MAX_EF = 256;
constant uint MAX_VISITED = 4096;
constant uint MAX_PROBES = 32;
constant uint EMPTY_SLOT = 0xFFFFFFFFu;

struct CandidateEntry {
    uint nodeID;
    float distance;
};

inline float compute_distance(
    device const float *vectors,
    device const float *query,
    uint nodeID,
    uint dim,
    uint metricType
) {
    uint base = nodeID * dim;

    if (metricType == 1) {
        float sumSq = 0.0f;
        for (uint d = 0; d < dim; d++) {
            float diff = query[d] - vectors[base + d];
            sumSq += diff * diff;
        }
        return sumSq;
    }

    float dot = 0.0f;
    if (metricType == 2) {
        for (uint d = 0; d < dim; d++) {
            dot += query[d] * vectors[base + d];
        }
        return -dot;
    }

    float normQSq = 0.0f;
    float normVSq = 0.0f;
    for (uint d = 0; d < dim; d++) {
        float q = query[d];
        float v = vectors[base + d];
        dot += q * v;
        normQSq += q * q;
        normVSq += v * v;
    }
    float denom = sqrt(normQSq) * sqrt(normVSq);
    return (denom < 1e-10f) ? 1.0f : (1.0f - (dot / denom));
}

inline bool try_visit(threadgroup atomic_uint *visited, uint nodeID) {
    uint hash = nodeID * 2654435761u;

    for (uint probe = 0; probe < MAX_PROBES; probe++) {
        uint slot = (hash + probe) & (MAX_VISITED - 1);
        uint expected = EMPTY_SLOT;
        if (atomic_compare_exchange_weak_explicit(
                &visited[slot],
                &expected,
                nodeID,
                memory_order_relaxed,
                memory_order_relaxed)) {
            return true;
        }

        if (expected == nodeID) {
            return false;
        }
    }

    return false;
}

inline void append_entry(
    threadgroup CandidateEntry *entries,
    threadgroup atomic_uint &count,
    uint limit,
    CandidateEntry entry
) {
    uint current = atomic_load_explicit(&count, memory_order_relaxed);
    while (current < limit) {
        uint expected = current;
        if (atomic_compare_exchange_weak_explicit(
                &count,
                &expected,
                current + 1,
                memory_order_relaxed,
                memory_order_relaxed)) {
            entries[current] = entry;
            return;
        }
        current = expected;
    }
}

inline void insertion_sort(
    threadgroup CandidateEntry *entries,
    uint start,
    uint end
) {
    if (end <= start + 1) {
        return;
    }

    for (uint i = start + 1; i < end; i++) {
        CandidateEntry key = entries[i];
        int j = int(i) - 1;
        while (j >= int(start) && key.distance < entries[j].distance) {
            entries[j + 1] = entries[j];
            j--;
        }
        entries[j + 1] = key;
    }
}

kernel void beam_search(
    device const float *vectors [[buffer(0)]],
    device const uint *adjacency [[buffer(1)]],
    device const float *query [[buffer(2)]],
    device float *output_dists [[buffer(3)]],
    device uint *output_ids [[buffer(4)]],
    constant uint &node_count [[buffer(5)]],
    constant uint &degree [[buffer(6)]],
    constant uint &dim [[buffer(7)]],
    constant uint &k [[buffer(8)]],
    constant uint &ef [[buffer(9)]],
    constant uint &entry_point [[buffer(10)]],
    constant uint &metric_type [[buffer(11)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup CandidateEntry candidates[MAX_EF];
    threadgroup CandidateEntry results[MAX_EF];
    threadgroup atomic_uint visited[MAX_VISITED];
    threadgroup atomic_uint candidate_count;
    threadgroup atomic_uint result_count;
    threadgroup uint candidate_head;
    threadgroup CandidateEntry current;
    threadgroup uint should_stop;

    uint ef_limit = min(min(ef, node_count), MAX_EF);
    uint output_k = min(k, ef_limit);

    for (uint index = tid; index < MAX_VISITED; index += threads_per_group) {
        atomic_store_explicit(&visited[index], EMPTY_SLOT, memory_order_relaxed);
    }

    if (tid == 0) {
        atomic_store_explicit(&candidate_count, 0, memory_order_relaxed);
        atomic_store_explicit(&result_count, 0, memory_order_relaxed);
        candidate_head = 0;
        should_stop = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0 && ef_limit > 0 && entry_point < node_count) {
        float entry_dist = compute_distance(vectors, query, entry_point, dim, metric_type);
        CandidateEntry entry = { entry_point, entry_dist };
        candidates[0] = entry;
        results[0] = entry;
        atomic_store_explicit(&candidate_count, 1, memory_order_relaxed);
        atomic_store_explicit(&result_count, 1, memory_order_relaxed);
        (void)try_visit(visited, entry_point);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint max_iterations = max((uint)1, ef_limit * 2);

    for (uint iteration = 0; iteration < max_iterations; iteration++) {
        if (tid == 0) {
            uint local_candidate_count = atomic_load_explicit(&candidate_count, memory_order_relaxed);
            uint local_result_count = atomic_load_explicit(&result_count, memory_order_relaxed);

            if (candidate_head >= local_candidate_count || candidate_head >= ef_limit) {
                should_stop = 1;
            } else {
                current = candidates[candidate_head];
                candidate_head += 1;
                should_stop = 0;

                if (local_result_count >= ef_limit && local_result_count > 0) {
                    float worst = results[local_result_count - 1].distance;
                    if (current.distance > worst) {
                        should_stop = 1;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (should_stop != 0) {
            break;
        }

        if (tid < degree) {
            uint neighbor_index = current.nodeID * degree + tid;
            uint neighbor_id = adjacency[neighbor_index];
            if (neighbor_id != EMPTY_SLOT && neighbor_id < node_count) {
                if (try_visit(visited, neighbor_id)) {
                    float dist = compute_distance(vectors, query, neighbor_id, dim, metric_type);
                    CandidateEntry next = { neighbor_id, dist };
                    append_entry(results, result_count, MAX_EF, next);
                    append_entry(candidates, candidate_count, MAX_EF, next);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            uint local_result_count = min(atomic_load_explicit(&result_count, memory_order_relaxed), MAX_EF);
            insertion_sort(results, 0, local_result_count);
            if (local_result_count > ef_limit) {
                local_result_count = ef_limit;
                atomic_store_explicit(&result_count, local_result_count, memory_order_relaxed);
            }

            uint local_candidate_count = min(atomic_load_explicit(&candidate_count, memory_order_relaxed), MAX_EF);
            uint active_start = min(candidate_head, local_candidate_count);
            insertion_sort(candidates, active_start, local_candidate_count);
            if (local_candidate_count - active_start > ef_limit) {
                local_candidate_count = active_start + ef_limit;
                atomic_store_explicit(&candidate_count, local_candidate_count, memory_order_relaxed);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        uint final_count = min(atomic_load_explicit(&result_count, memory_order_relaxed), ef_limit);
        uint write_count = min(output_k, final_count);

        for (uint i = 0; i < write_count; i++) {
            output_ids[i] = results[i].nodeID;
            output_dists[i] = results[i].distance;
        }

        for (uint i = write_count; i < k; i++) {
            output_ids[i] = EMPTY_SLOT;
            output_dists[i] = FLT_MAX;
        }
    }
}
