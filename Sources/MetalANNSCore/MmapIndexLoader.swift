import Foundation
import Metal
import Darwin

public enum MmapIndexLoader {
    private static let headerMagic: [UInt8] = [0x4D, 0x41, 0x4E, 0x4E] // "MANN"
    private static let mmapVersion: UInt32 = 3
    private static var pageSize: Int { max(1, Int(getpagesize())) }

    public struct MmapLoadResult {
        public let vectors: any VectorStorage
        public let graph: GraphBuffer
        public let idMap: IDMap
        public let entryPoint: UInt32
        public let metric: Metric
        public let mmapLifetime: AnyObject

        public init(
            vectors: any VectorStorage,
            graph: GraphBuffer,
            idMap: IDMap,
            entryPoint: UInt32,
            metric: Metric,
            mmapLifetime: AnyObject
        ) {
            self.vectors = vectors
            self.graph = graph
            self.idMap = idMap
            self.entryPoint = entryPoint
            self.metric = metric
            self.mmapLifetime = mmapLifetime
        }
    }

    public static func load(from fileURL: URL, device: MTLDevice? = nil) throws -> MmapLoadResult {
        let region = try MmapRegion(fileURL: fileURL)

        guard let metalDevice = device ?? MTLCreateSystemDefaultDevice() else {
            throw ANNSError.constructionFailed("No Metal device available")
        }

        var cursor = 0
        let magic = try readBytes(from: region.pointer, length: region.length, offset: cursor, count: 4)
        cursor += 4
        guard magic == headerMagic else {
            throw ANNSError.corruptFile("Invalid file magic")
        }

        let formatVersion = try readUInt32(from: region.pointer, length: region.length, cursor: &cursor)
        guard formatVersion == mmapVersion else {
            throw ANNSError.corruptFile("Mmap loader requires version \(mmapVersion), found \(formatVersion)")
        }

        let nodeCount = Int(try readUInt32(from: region.pointer, length: region.length, cursor: &cursor))
        let degree = Int(try readUInt32(from: region.pointer, length: region.length, cursor: &cursor))
        let dim = Int(try readUInt32(from: region.pointer, length: region.length, cursor: &cursor))
        let metricCode = try readUInt32(from: region.pointer, length: region.length, cursor: &cursor)
        let metric = try metric(from: metricCode)
        let storageType = try readUInt32(from: region.pointer, length: region.length, cursor: &cursor)

        guard nodeCount > 0 else {
            throw ANNSError.corruptFile("Node count must be greater than zero")
        }
        guard degree > 0 else {
            throw ANNSError.corruptFile("Degree must be greater than zero")
        }
        guard storageType == 0 || storageType == 1 else {
            throw ANNSError.corruptFile("Unsupported storage type \(storageType)")
        }

        let bytesPerElement = storageType == 1 ? MemoryLayout<UInt16>.stride : MemoryLayout<Float>.stride
        let vectorByteCount = nodeCount * dim * bytesPerElement
        let adjacencyByteCount = nodeCount * degree * MemoryLayout<UInt32>.stride
        let distanceByteCount = nodeCount * degree * MemoryLayout<Float>.stride

        let alignment = pageSize
        let vectorOffset = alignedOffset(cursor, alignment: alignment)
        let adjacencyOffset = alignedOffset(vectorOffset + vectorByteCount, alignment: alignment)
        let distanceOffset = alignedOffset(adjacencyOffset + adjacencyByteCount, alignment: alignment)
        let trailerOffset = alignedOffset(distanceOffset + distanceByteCount, alignment: alignment)

        let vectorMappedLength = alignedLength(vectorByteCount, alignment: alignment)
        let adjacencyMappedLength = alignedLength(adjacencyByteCount, alignment: alignment)
        let distanceMappedLength = alignedLength(distanceByteCount, alignment: alignment)

        guard vectorOffset + vectorMappedLength <= region.length,
              adjacencyOffset + adjacencyMappedLength <= region.length,
              distanceOffset + distanceMappedLength <= region.length,
              trailerOffset + MemoryLayout<UInt32>.size <= region.length else {
            throw ANNSError.corruptFile("Mmap sections are truncated")
        }

        let vectorPointer = region.pointer.advanced(by: vectorOffset)
        let adjacencyPointer = region.pointer.advanced(by: adjacencyOffset)
        let distancePointer = region.pointer.advanced(by: distanceOffset)

        guard let vectorBuffer = metalDevice.makeBuffer(
            bytesNoCopy: vectorPointer,
            length: vectorMappedLength,
            options: .storageModeShared,
            deallocator: nil
        ) else {
            throw ANNSError.constructionFailed("Failed to create mmap vector buffer")
        }
        guard let adjacencyBuffer = metalDevice.makeBuffer(
            bytesNoCopy: adjacencyPointer,
            length: adjacencyMappedLength,
            options: .storageModeShared,
            deallocator: nil
        ) else {
            throw ANNSError.constructionFailed("Failed to create mmap adjacency buffer")
        }
        guard let distanceBuffer = metalDevice.makeBuffer(
            bytesNoCopy: distancePointer,
            length: distanceMappedLength,
            options: .storageModeShared,
            deallocator: nil
        ) else {
            throw ANNSError.constructionFailed("Failed to create mmap distance buffer")
        }

        let vectors = MmapVectorStorage(
            buffer: vectorBuffer,
            dim: dim,
            count: nodeCount,
            isFloat16: storageType == 1
        )
        let graph = try GraphBuffer(
            adjacencyBuffer: adjacencyBuffer,
            distanceBuffer: distanceBuffer,
            capacity: nodeCount,
            degree: degree,
            nodeCount: nodeCount
        )

        var trailerCursor = trailerOffset
        let idMapByteCount = Int(try readUInt32(from: region.pointer, length: region.length, cursor: &trailerCursor))
        guard trailerCursor + idMapByteCount + MemoryLayout<UInt32>.size <= region.length else {
            throw ANNSError.corruptFile("Truncated IDMap payload")
        }

        let idMapData = Data(bytes: region.pointer.advanced(by: trailerCursor), count: idMapByteCount)
        trailerCursor += idMapByteCount
        let entryPoint = try readUInt32(from: region.pointer, length: region.length, cursor: &trailerCursor)

        let idMap: IDMap
        do {
            idMap = try JSONDecoder().decode(IDMap.self, from: idMapData)
        } catch {
            throw ANNSError.corruptFile("IDMap payload is corrupt")
        }

        return MmapLoadResult(
            vectors: vectors,
            graph: graph,
            idMap: idMap,
            entryPoint: entryPoint,
            metric: metric,
            mmapLifetime: region
        )
    }

    private static func metric(from code: UInt32) throws -> Metric {
        switch code {
        case 0:
            return .cosine
        case 1:
            return .l2
        case 2:
            return .innerProduct
        default:
            throw ANNSError.corruptFile("Unsupported metric code")
        }
    }

    private static func readBytes(
        from pointer: UnsafeMutableRawPointer,
        length: Int,
        offset: Int,
        count: Int
    ) throws -> [UInt8] {
        guard offset >= 0, count >= 0, offset + count <= length else {
            throw ANNSError.corruptFile("Unexpected EOF")
        }
        let base = pointer.bindMemory(to: UInt8.self, capacity: length)
        return Array(UnsafeBufferPointer(start: base.advanced(by: offset), count: count))
    }

    private static func readUInt32(
        from pointer: UnsafeMutableRawPointer,
        length: Int,
        cursor: inout Int
    ) throws -> UInt32 {
        guard cursor + MemoryLayout<UInt32>.size <= length else {
            throw ANNSError.corruptFile("Unexpected EOF")
        }
        let base = pointer.bindMemory(to: UInt8.self, capacity: length).advanced(by: cursor)
        let value = UInt32(base[0]) |
            (UInt32(base[1]) << 8) |
            (UInt32(base[2]) << 16) |
            (UInt32(base[3]) << 24)
        cursor += MemoryLayout<UInt32>.size
        return value
    }

    private static func alignedOffset(_ value: Int, alignment: Int) -> Int {
        let remainder = value % alignment
        if remainder == 0 {
            return value
        }
        return value + (alignment - remainder)
    }

    private static func alignedLength(_ value: Int, alignment: Int) -> Int {
        let remainder = value % alignment
        if remainder == 0 {
            return value
        }
        return value + (alignment - remainder)
    }
}

private final class MmapRegion: @unchecked Sendable {
    let pointer: UnsafeMutableRawPointer
    let length: Int
    private let descriptor: Int32

    init(fileURL: URL) throws {
        let fd = open(fileURL.path, O_RDONLY)
        guard fd >= 0 else {
            throw ANNSError.corruptFile("Unable to open mmap file")
        }

        var fileStat = stat()
        guard fstat(fd, &fileStat) == 0 else {
            _ = close(fd)
            throw ANNSError.corruptFile("Unable to stat mmap file")
        }

        let mappedLength = Int(fileStat.st_size)
        guard mappedLength > 0 else {
            _ = close(fd)
            throw ANNSError.corruptFile("Mmap file is empty")
        }

        let mappedPointer = mmap(nil, mappedLength, PROT_READ, MAP_PRIVATE, fd, 0)
        guard mappedPointer != MAP_FAILED else {
            _ = close(fd)
            throw ANNSError.corruptFile("mmap failed for file")
        }

        self.pointer = mappedPointer!
        self.length = mappedLength
        self.descriptor = fd
    }

    deinit {
        _ = munmap(pointer, length)
        _ = close(descriptor)
    }
}

private final class MmapVectorStorage: VectorStorage, @unchecked Sendable {
    let buffer: MTLBuffer
    let dim: Int
    let capacity: Int
    private(set) var count: Int
    let isFloat16: Bool

    init(buffer: MTLBuffer, dim: Int, count: Int, isFloat16: Bool) {
        self.buffer = buffer
        self.dim = dim
        self.capacity = count
        self.count = count
        self.isFloat16 = isFloat16
    }

    func setCount(_ newCount: Int) {
        count = min(max(0, newCount), capacity)
    }

    func insert(vector: [Float], at index: Int) throws {
        throw ANNSError.constructionFailed("Mmap-loaded vectors are read-only")
    }

    func batchInsert(vectors: [[Float]], startingAt start: Int) throws {
        throw ANNSError.constructionFailed("Mmap-loaded vectors are read-only")
    }

    func vector(at index: Int) -> [Float] {
        precondition(index >= 0 && index < count, "Index out of bounds")

        if isFloat16 {
            let pointer = buffer.contents().bindMemory(to: UInt16.self, capacity: max(1, capacity * dim))
            let base = index * dim
            var result = [Float](repeating: 0, count: dim)
            for dimIndex in 0..<dim {
                result[dimIndex] = Float(Float16(bitPattern: pointer[base + dimIndex]))
            }
            return result
        } else {
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: max(1, capacity * dim))
            let base = index * dim
            return Array(UnsafeBufferPointer(start: pointer.advanced(by: base), count: dim))
        }
    }
}
