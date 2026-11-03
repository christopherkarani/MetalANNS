import Foundation
import MetalANNSCore

/// External key of an inserted vector: string ids (`insert(_:id:)`, and every
/// `VectorIndex` key) or numeric ids (`insert(_:numericID:)`).
private enum InsertKey {
    case string(String)
    case numeric(UInt64)

    var description: String {
        switch self {
        case .string(let id): id
        case .numeric(let id): String(id)
        }
    }

    func internalID(in idMap: IDMap) -> UInt32? {
        switch self {
        case .string(let id): idMap.internalID(for: id)
        case .numeric(let id): idMap.internalID(forNumeric: id)
        }
    }

    func assign(in idMap: inout IDMap) -> UInt32? {
        switch self {
        case .string(let id): idMap.assign(externalID: id)
        case .numeric(let id): idMap.assign(numericID: id)
        }
    }
}

extension GraphIndex {
    /// Inserts one vector.
    ///
    /// - An id that was deleted (and not yet compacted away) is re-inserted
    ///   into its old slot with the new vector.
    /// - Storage grows geometrically when full; there is no fixed capacity.
    /// - Throws `ANNSError.idAlreadyExists` for a live id and
    ///   `ANNSError.nonFiniteVector` for NaN/±inf components.
    public func insert(_ vector: [Float], id: String) async throws {
        try await insertOne(vector, key: .string(id))
    }

    /// Inserts a vector with a numeric (UInt64) key. Same semantics as
    /// `insert(_:id:)`.
    public func insert(_ vector: [Float], numericID: UInt64) async throws {
        try await insertOne(vector, key: .numeric(numericID))
    }

    private func insertOne(_ vector: [Float], key: InsertKey) async throws {
        await beginWrite()
        defer { endWrite() }

        let vectors = try requireMutableIndex().vectors
        guard vector.count == vectors.dim else {
            throw ANNSError.dimensionMismatch(expected: vectors.dim, got: vector.count)
        }
        try VectorValidation.requireFinite(vector)

        let metricsRecorder = metrics
        let insertStart = metricsRecorder == nil ? nil : ContinuousClock.now

        if let existing = key.internalID(in: idMap) {
            guard softDeletion.isDeleted(existing) else {
                throw ANNSError.idAlreadyExists(key.description)
            }
            try reinsert(vector, at: Int(existing))
        } else {
            guard idMap.canAllocate(1) else {
                throw ANNSError.constructionFailed("Internal ID space exhausted")
            }
            let slot = Int(idMap.nextInternalID)
            try ensureCapacity(slot + 1)
            try appendNode(vector, at: slot, key: key)
        }

        if let metricsRecorder, let insertStart {
            let duration = ContinuousClock.now - insertStart
            await metricsRecorder.recordInsert(durationNs: Self.durationNanoseconds(duration))
        }
    }

    /// Inserts a batch. Ids must be unique within the batch; ids that were
    /// deleted are re-inserted into their old slots, the rest are appended.
    /// New rows are appended atomically (all or none); re-inserts are applied
    /// after them, one by one.
    public func batchInsert(_ vectors: [[Float]], ids: [String]) async throws {
        await beginWrite()
        defer { endWrite() }

        let vectorStorage = try requireMutableIndex().vectors
        guard vectors.count == ids.count else {
            throw ANNSError.constructionFailed("Vector and ID counts do not match")
        }
        guard !vectors.isEmpty else {
            return
        }

        let dim = vectorStorage.dim
        for vector in vectors {
            guard vector.count == dim else {
                throw ANNSError.dimensionMismatch(expected: dim, got: vector.count)
            }
        }
        try VectorValidation.requireFinite(rows: vectors)

        var seenIDs = Set<String>()
        var appendOffsets: [Int] = []
        var reinserts: [(offset: Int, slot: Int)] = []
        for (offset, id) in ids.enumerated() {
            if !seenIDs.insert(id).inserted {
                throw ANNSError.idAlreadyExists(id)
            }
            if let existing = idMap.internalID(for: id) {
                guard softDeletion.isDeleted(existing) else {
                    throw ANNSError.idAlreadyExists(id)
                }
                reinserts.append((offset, Int(existing)))
            } else {
                appendOffsets.append(offset)
            }
        }

        let metricsRecorder = metrics
        let batchInsertStart = metricsRecorder == nil ? nil : ContinuousClock.now

        if !appendOffsets.isEmpty {
            guard idMap.canAllocate(appendOffsets.count) else {
                throw ANNSError.constructionFailed("Internal ID space exhausted")
            }
            let startSlot = Int(idMap.nextInternalID)
            try ensureCapacity(startSlot + appendOffsets.count)
            try appendBatch(
                appendOffsets.map { vectors[$0] },
                ids: appendOffsets.map { ids[$0] },
                startingAt: startSlot
            )
        }
        for (offset, slot) in reinserts {
            try reinsert(vectors[offset], at: slot)
        }

        if let metricsRecorder, let batchInsertStart {
            let duration = ContinuousClock.now - batchInsertStart
            await metricsRecorder.recordBatchInsert(
                count: vectors.count,
                durationNs: Self.durationNanoseconds(duration)
            )
        }
    }

    private func requireMutableIndex() throws -> (vectors: any VectorStorage, graph: GraphBuffer) {
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        return (vectors, graph)
    }

    /// Grows vector and graph storage so `required` rows fit. Doubles capacity,
    /// so a run of single inserts costs amortized O(1) copies per row.
    internal func ensureCapacity(_ required: Int) throws {
        guard let vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        guard required > vectors.capacity || required > graph.capacity else {
            return
        }
        let newCapacity = StorageGrowth.grownCapacity(
            current: min(vectors.capacity, graph.capacity),
            required: required
        )
        let device = context?.device
        if vectors.capacity < newCapacity {
            guard let grown = try StorageGrowth.grow(vectors, to: newCapacity, device: device) else {
                throw ANNSError.indexCapacityExceeded
            }
            invalidateSearchCaches(for: vectors)
            self.vectors = grown
        }
        if graph.capacity < newCapacity {
            self.graph = try StorageGrowth.grow(graph, to: newCapacity, device: device)
        }
    }

    /// Drops process-wide search caches keyed by a buffer that is being replaced,
    /// so they neither pin its memory nor hit if its address is reused.
    private func invalidateSearchCaches(for storage: any VectorStorage) {
        StorageGrowth.releaseSearchCaches(for: storage.buffer)
    }

    /// Appends a fresh node at `slot` (== `idMap.nextInternalID`); rolls back on failure.
    private func appendNode(_ vector: [Float], at slot: Int, key: InsertKey) throws {
        guard let vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        let graphVector = vectors is BinaryVectorBuffer ? Self.quantizeForHamming(vector) : vector
        let previousVectorCount = vectors.count
        let previousGraphCount = graph.nodeCount

        do {
            try vectors.insert(vector: graphVector, at: slot)
            if vectors.count < slot + 1 {
                vectors.setCount(slot + 1)
            }

            try IncrementalBuilder.insert(
                vector: graphVector,
                at: slot,
                into: graph,
                vectors: vectors,
                entryPoint: entryPoint,
                metric: configuration.metric,
                degree: configuration.degree
            )
            if graph.nodeCount < slot + 1 {
                graph.setCount(slot + 1)
            }
            hnsw = nil

            try scheduleRepair(slots: [slot], immediately: false)

            guard let assignedID = key.assign(in: &idMap), Int(assignedID) == slot else {
                throw ANNSError.constructionFailed("Failed to commit internal ID for '\(key.description)'")
            }
        } catch {
            vectors.setCount(previousVectorCount)
            graph.setCount(previousGraphCount)

            // Best-effort cleanup of the uncommitted node slot.
            let emptyIDs = Array(repeating: UInt32.max, count: configuration.degree)
            let emptyDistances = Array(repeating: Float.greatestFiniteMagnitude, count: configuration.degree)
            try? graph.setNeighbors(of: slot, ids: emptyIDs, distances: emptyDistances)

            if let annError = error as? ANNSError {
                throw annError
            }
            throw ANNSError.constructionFailed("Incremental insert failed: \(error)")
        }
    }

    /// Appends fresh nodes at `startSlot...`; all or nothing.
    private func appendBatch(_ vectors: [[Float]], ids: [String], startingAt startSlot: Int) throws {
        guard let vectorStorage = self.vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        let slots = Array(startSlot..<(startSlot + vectors.count))
        let previousVectorCount = vectorStorage.count
        let previousGraphCount = graph.nodeCount

        do {
            let graphVectors: [[Float]]
            if vectorStorage is BinaryVectorBuffer {
                graphVectors = vectors.map(Self.quantizeForHamming)
            } else {
                graphVectors = vectors
            }

            for (offset, vector) in graphVectors.enumerated() {
                try vectorStorage.insert(vector: vector, at: slots[offset])
            }
            let newMaxCount = (slots.last ?? 0) + 1
            if vectorStorage.count < newMaxCount {
                vectorStorage.setCount(newMaxCount)
            }

            try BatchIncrementalBuilder.batchInsert(
                vectors: graphVectors,
                startingAt: startSlot,
                into: graph,
                vectorStorage: vectorStorage,
                entryPoint: entryPoint,
                metric: configuration.metric,
                degree: configuration.degree
            )

            if graph.nodeCount < newMaxCount {
                graph.setCount(newMaxCount)
            }
            hnsw = nil

            try scheduleRepair(slots: slots, immediately: true)

            for (offset, id) in ids.enumerated() {
                guard let assignedID = idMap.assign(externalID: id), Int(assignedID) == slots[offset] else {
                    throw ANNSError.constructionFailed("Failed to commit internal ID for '\(id)'")
                }
            }
        } catch {
            vectorStorage.setCount(previousVectorCount)
            graph.setCount(previousGraphCount)
            pendingRepairIDs.removeAll { internalID in
                slots.contains(Int(internalID))
            }

            // Best-effort cleanup for uncommitted slots.
            let emptyIDs = Array(repeating: UInt32.max, count: configuration.degree)
            let emptyDistances = Array(repeating: Float.greatestFiniteMagnitude, count: configuration.degree)
            for slot in slots {
                try? graph.setNeighbors(of: slot, ids: emptyIDs, distances: emptyDistances)
            }

            if let annError = error as? ANNSError {
                throw annError
            }
            throw ANNSError.constructionFailed("Batch insert failed: \(error)")
        }
    }

    /// Re-inserts a soft-deleted id into its own slot with a new vector.
    ///
    /// The slot keeps its internal ID (so the id map, persistence layout and
    /// node count are unchanged). Its out-edges are recomputed from the new
    /// vector; every edge pointing at it gets a fresh distance so later inserts
    /// evict them correctly. The slot is undeleted only after the
    /// graph update succeeds, so a failure leaves the id deleted.
    private func reinsert(_ vector: [Float], at slot: Int) throws {
        guard let vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        let graphVector = vectors is BinaryVectorBuffer ? Self.quantizeForHamming(vector) : vector

        do {
            try vectors.insert(vector: graphVector, at: slot)
            try refreshInEdges(to: slot, vector: graphVector)
            try IncrementalBuilder.insert(
                vector: graphVector,
                at: slot,
                into: graph,
                vectors: vectors,
                entryPoint: entryPoint,
                metric: configuration.metric,
                degree: configuration.degree
            )
            hnsw = nil
            try scheduleRepair(slots: [slot], immediately: false)
        } catch {
            if let annError = error as? ANNSError {
                throw annError
            }
            throw ANNSError.constructionFailed("Re-insert failed: \(error)")
        }
        softDeletion.undelete(UInt32(slot))
    }

    /// Recomputes the stored distance on every edge that points at `slot`.
    ///
    /// Edges carry the distance to their target, and the slot's vector just
    /// changed. Stale (usually too-small) distances would keep a now-distant
    /// node in its in-neighbours' lists and skew their eviction order. In-edges
    /// are not indexed, so this is one pass over the adjacency (O(n·degree)
    /// reads), cheap next to the insert's own graph search.
    private func refreshInEdges(to slot: Int, vector: [Float]) throws {
        guard let vectors, let graph else {
            return
        }
        let target = UInt32(slot)
        let degree = graph.degree
        let adjacency = graph.adjacencyBuffer.contents().bindMemory(
            to: UInt32.self,
            capacity: graph.capacity * degree
        )
        for node in 0..<graph.nodeCount where node != slot {
            let row = adjacency + node * degree
            guard (0..<degree).contains(where: { row[$0] == target }) else { continue }
            var ids = graph.neighborIDs(of: node)
            var distances = graph.neighborDistances(of: node)
            let distance = SIMDDistance.distance(vectors.vector(at: node), vector, metric: configuration.metric)
            for position in 0..<degree where ids[position] == target {
                distances[position] = distance
            }
            let sorted = zip(ids, distances).sorted { $0.1 < $1.1 }
            ids = sorted.map(\.0)
            distances = sorted.map(\.1)
            try graph.setNeighbors(of: node, ids: ids, distances: distances)
        }
    }

    /// Queues `slots` for graph repair. Single inserts repair every
    /// `repairInterval` nodes; batches repair immediately (as before).
    private func scheduleRepair(slots: [Int], immediately: Bool) throws(ANNSError) {
        let repairConfig = configuration.repairConfiguration
        guard repairConfig.enabled else { return }
        if immediately {
            pendingRepairIDs.append(contentsOf: slots.map(UInt32.init))
            try triggerRepair(throwOnFailure: false)
        } else if repairConfig.repairInterval > 0 {
            pendingRepairIDs.append(contentsOf: slots.map(UInt32.init))
            if pendingRepairIDs.count >= repairConfig.repairInterval {
                try triggerRepair(throwOnFailure: false)
            }
        }
    }

    public func repair() async throws(ANNSError) {
        await beginWrite()
        defer { endWrite() }
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard isBuilt, vectors != nil, graph != nil else {
            throw ANNSError.indexEmpty
        }
        guard !pendingRepairIDs.isEmpty else {
            return
        }
        try triggerRepair()
    }

    private func triggerRepair(throwOnFailure: Bool = true) throws(ANNSError) {
        guard let vectors, let graph else {
            return
        }
        guard !pendingRepairIDs.isEmpty else {
            return
        }

        let idsToRepair = pendingRepairIDs
        pendingRepairIDs.removeAll(keepingCapacity: true)

        do {
            _ = try GraphRepairer.repair(
                recentIDs: idsToRepair,
                vectors: vectors,
                graph: graph,
                config: configuration.repairConfiguration,
                metric: configuration.metric
            )
            hnsw = nil
        } catch {
            // Preserve pending IDs so repair can be retried later.
            pendingRepairIDs = idsToRepair + pendingRepairIDs
            if throwOnFailure {
                throw error
            }
        }
    }

    public func delete(id: String) throws {
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard let internalID = idMap.internalID(for: id) else {
            throw ANNSError.idNotFound(id)
        }
        softDeletion.markDeleted(internalID)
        metadataStore.remove(id: internalID)
    }

    /// Soft-deletes a vector inserted with `insert(_:numericID:)`.
    public func delete(numericID: UInt64) throws {
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard let internalID = idMap.internalID(forNumeric: numericID) else {
            throw ANNSError.idNotFound(String(numericID))
        }
        softDeletion.markDeleted(internalID)
        metadataStore.remove(id: internalID)
    }

    /// Rebuilds the graph without soft-deleted nodes and releases their slots
    /// (and ids). Exclusive: inserts and searches issued meanwhile wait.
    public func compact() async throws {
        await beginWrite()
        defer { endWrite() }
        guard !isReadOnlyLoadedIndex else {
            throw ANNSError.constructionFailed("Index is read-only (mmap-loaded)")
        }
        guard isBuilt, let vectors, let graph else {
            throw ANNSError.indexEmpty
        }
        guard softDeletion.deletedCount > 0 else {
            return
        }

        // `delete` and `setMetadata` are synchronous and ungated, so they can
        // run while the compactor is suspended. Compact against this snapshot,
        // then carry later deletes over to the new slots.
        let compactedDeletion = softDeletion
        let result = try await IndexCompactor.compact(
            vectors: vectors,
            graph: graph,
            idMap: idMap,
            softDeletion: compactedDeletion,
            metric: configuration.metric,
            degree: configuration.degree,
            context: context,
            maxIterations: configuration.maxIterations,
            convergenceThreshold: configuration.convergenceThreshold,
            useFloat16: configuration.useFloat16,
            useBinary: configuration.useBinary
        )

        var remapping: [UInt32: UInt32] = [:]
        remapping.reserveCapacity(result.idMap.count)
        for oldIndex in 0..<vectors.count {
            let oldID = UInt32(oldIndex)
            if compactedDeletion.isDeleted(oldID) {
                continue
            }
            let newID: UInt32?
            if let externalID = idMap.externalID(for: oldID) {
                newID = result.idMap.internalID(for: externalID)
            } else if let numericID = idMap.numericID(for: oldID) {
                newID = result.idMap.internalID(forNumeric: numericID)
            } else {
                newID = nil
            }
            guard let newID else {
                continue
            }
            remapping[oldID] = newID
        }
        var deletedDuringCompaction = SoftDeletion()
        for (oldID, newID) in remapping where softDeletion.isDeleted(oldID) {
            deletedDuringCompaction.markDeleted(newID)
        }

        invalidateSearchCaches(for: vectors)
        self.vectors = result.vectors
        self.graph = result.graph
        self.idMap = result.idMap
        self.entryPoint = result.entryPoint
        self.softDeletion = deletedDuringCompaction
        self.metadataStore = metadataStore.remapped(using: remapping)
        self.isReadOnlyLoadedIndex = false
        self.mmapLifetime = nil
        self.pendingRepairIDs.removeAll()
        self.hnsw = nil
    }

    public func setMetadata(_ column: String, value: String, for id: String) throws {
        guard isBuilt else {
            throw ANNSError.indexEmpty
        }
        guard let internalID = idMap.internalID(for: id) else {
            throw ANNSError.idNotFound(id)
        }
        metadataStore.set(column, stringValue: value, for: internalID)
    }

    public func setMetadata(_ column: String, value: Float, for id: String) throws {
        guard isBuilt else {
            throw ANNSError.indexEmpty
        }
        guard let internalID = idMap.internalID(for: id) else {
            throw ANNSError.idNotFound(id)
        }
        metadataStore.set(column, floatValue: value, for: internalID)
    }

    public func setMetadata(_ column: String, value: Int64, for id: String) throws {
        guard isBuilt else {
            throw ANNSError.indexEmpty
        }
        guard let internalID = idMap.internalID(for: id) else {
            throw ANNSError.idNotFound(id)
        }
        metadataStore.set(column, intValue: value, for: internalID)
    }

}
