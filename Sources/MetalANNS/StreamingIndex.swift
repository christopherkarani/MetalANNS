import Foundation
import MetalANNSCore

/// A continuous-ingest index that uses a two-level merge architecture.
///
/// New vectors land in a small `delta` index. When the delta reaches
/// `StreamingConfiguration.deltaCapacity` it is merged into the frozen
/// `base` index asynchronously (or synchronously for `.blocking` strategy).
public actor StreamingIndex {
    private var base: ANNSIndex?
    private var mergeTask: Task<Void, Error>?
    private var _isMerging = false

    private var pendingVectors: [[Float]] = []
    private var pendingIDs: [String] = []
    private var delta: ANNSIndex?

    private var allVectorsList: [[Float]] = []
    private var allIDsList: [String] = []
    private var allIDs: Set<String> = []
    private var deletedIDs: Set<String> = []

    private var idInBase: Set<String> = []
    private var idInDelta: Set<String> = []

    private var vectorDimension: Int?

    private let config: StreamingConfiguration

    public init(config: StreamingConfiguration = .default) {
        self.config = config
    }

    public var count: Int {
        allIDsList.count - deletedIDs.count
    }

    public var isMerging: Bool {
        _isMerging
    }

    public func insert(_ vector: [Float], id: String) async throws {
        guard !allIDs.contains(id) else {
            throw ANNSError.idAlreadyExists(id)
        }
        try validateDimension(of: vector)

        allIDs.insert(id)
        allIDsList.append(id)
        allVectorsList.append(vector)
        pendingIDs.append(id)
        pendingVectors.append(vector)

        try await flushPendingIntoDelta()
        try await maybeTriggerMerge()
    }

    public func batchInsert(_ vectors: [[Float]], ids: [String]) async throws {
        guard vectors.count == ids.count else {
            throw ANNSError.constructionFailed("Vector and ID counts do not match")
        }
        guard !vectors.isEmpty else {
            return
        }

        var seen = Set<String>()
        for id in ids {
            if !seen.insert(id).inserted || allIDs.contains(id) {
                throw ANNSError.idAlreadyExists(id)
            }
        }

        for vector in vectors {
            try validateDimension(of: vector)
        }

        for id in ids {
            allIDs.insert(id)
            allIDsList.append(id)
        }
        allVectorsList.append(contentsOf: vectors)
        pendingIDs.append(contentsOf: ids)
        pendingVectors.append(contentsOf: vectors)

        try await flushPendingIntoDelta()
        try await maybeTriggerMerge()
    }

    private func validateDimension(of vector: [Float]) throws {
        if vector.isEmpty {
            throw ANNSError.dimensionMismatch(expected: 1, got: 0)
        }

        if let expected = vectorDimension {
            guard vector.count == expected else {
                throw ANNSError.dimensionMismatch(expected: expected, got: vector.count)
            }
        } else {
            vectorDimension = vector.count
        }
    }

    private func flushPendingIntoDelta() async throws {
        guard !pendingVectors.isEmpty else {
            return
        }

        if delta == nil {
            guard pendingVectors.count >= 2 else {
                return
            }
            let newDelta = ANNSIndex(configuration: adjustedConfiguration(for: pendingVectors.count))
            try await newDelta.build(vectors: pendingVectors, ids: pendingIDs)
            delta = newDelta
            idInDelta.formUnion(pendingIDs)
            pendingVectors.removeAll(keepingCapacity: true)
            pendingIDs.removeAll(keepingCapacity: true)
            return
        }

        do {
            try await delta?.batchInsert(pendingVectors, ids: pendingIDs)
            idInDelta.formUnion(pendingIDs)
            pendingVectors.removeAll(keepingCapacity: true)
            pendingIDs.removeAll(keepingCapacity: true)
        } catch let error as ANNSError {
            guard case .constructionFailed(let message) = error,
                  message.contains("Index capacity exceeded")
            else {
                throw error
            }
            try await triggerMerge()
            idInDelta.subtract(pendingIDs)
            pendingVectors.removeAll(keepingCapacity: true)
            pendingIDs.removeAll(keepingCapacity: true)
        }
    }

    private func shouldMerge() async -> Bool {
        guard let delta else {
            return false
        }
        return await delta.count >= config.deltaCapacity
    }

    private func maybeTriggerMerge() async throws {
        guard await shouldMerge() else {
            return
        }

        switch config.mergeStrategy {
        case .blocking:
            try await triggerMerge()
        case .background:
            guard mergeTask == nil, !_isMerging else {
                return
            }
            mergeTask = Task { [self] in
                defer {
                    Task { self.clearMergeTask() }
                }
                try await self.triggerMerge()
            }
        }
    }

    private func clearMergeTask() {
        mergeTask = nil
    }

    private func adjustedConfiguration(for nodeCount: Int) -> IndexConfiguration {
        var adjusted = config.indexConfiguration
        adjusted.degree = min(adjusted.degree, max(1, nodeCount - 1))
        return adjusted
    }

    private func activeRecords() -> (vectors: [[Float]], ids: [String]) {
        var vectors: [[Float]] = []
        var ids: [String] = []
        vectors.reserveCapacity(allVectorsList.count)
        ids.reserveCapacity(allIDsList.count)

        for (vector, id) in zip(allVectorsList, allIDsList) where !deletedIDs.contains(id) {
            vectors.append(vector)
            ids.append(id)
        }
        return (vectors, ids)
    }

    private func triggerMerge() async throws {
        guard !_isMerging else {
            return
        }
        _isMerging = true
        defer { _isMerging = false }

        let merged = activeRecords()
        guard !merged.vectors.isEmpty else {
            base = nil
            delta = nil
            idInBase.removeAll()
            idInDelta.removeAll()
            return
        }

        guard merged.vectors.count >= 2 else {
            base = nil
            delta = nil
            idInBase.removeAll()
            idInDelta.removeAll()
            return
        }

        let newBase = ANNSIndex(configuration: adjustedConfiguration(for: merged.vectors.count))
        try await newBase.build(vectors: merged.vectors, ids: merged.ids)

        base = newBase
        delta = nil
        idInBase = Set(merged.ids)
        idInDelta.removeAll()
    }
}
