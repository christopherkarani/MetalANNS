import Foundation
import Metal

public enum IndexSerializer {
    private static let headerMagic: [UInt8] = [0x4D, 0x41, 0x4E, 0x4E] // "MANN"
    private static let version: UInt32 = 1

    public static func save(
        vectors: VectorBuffer,
        graph: GraphBuffer,
        idMap: IDMap,
        entryPoint: UInt32,
        metric: Metric,
        to fileURL: URL
    ) throws {
        let nodeCount = vectors.count
        let degree = graph.degree

        guard nodeCount > 0 else {
            throw ANNSError.constructionFailed("Cannot save empty index")
        }
        guard nodeCount == graph.nodeCount else {
            throw ANNSError.constructionFailed("Vector and graph node counts do not match")
        }
        let vectorByteCount = nodeCount * vectors.dim * MemoryLayout<Float>.stride
        let adjacencyByteCount = nodeCount * degree * MemoryLayout<UInt32>.stride
        let distanceByteCount = nodeCount * degree * MemoryLayout<Float>.stride

        var filePayload = Data()
        append(magic: headerMagic, to: &filePayload)
        append(uint32: version, to: &filePayload)
        append(uint32: UInt32(nodeCount), to: &filePayload)
        append(uint32: UInt32(degree), to: &filePayload)
        append(uint32: UInt32(vectors.dim), to: &filePayload)
        append(uint32: metricCode(metric), to: &filePayload)

        filePayload.append(Data(bytes: vectors.buffer.contents(), count: vectorByteCount))
        filePayload.append(Data(bytes: graph.adjacencyBuffer.contents(), count: adjacencyByteCount))
        filePayload.append(Data(bytes: graph.distanceBuffer.contents(), count: distanceByteCount))

        let idMapData = try JSONEncoder().encode(idMap)
        append(uint32: UInt32(idMapData.count), to: &filePayload)
        filePayload.append(idMapData)
        append(uint32: entryPoint, to: &filePayload)

        try FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try filePayload.write(to: fileURL)
    }

    public static func load(from fileURL: URL, device: MTLDevice? = nil) throws -> (
        vectors: VectorBuffer,
        graph: GraphBuffer,
        idMap: IDMap,
        entryPoint: UInt32,
        metric: Metric
    ) {
        let payload = try Data(contentsOf: fileURL)

        guard payload.count >= 24 else {
            throw ANNSError.corruptFile("File is too small")
        }

        var cursor = 0
        let magic = Array(payload[cursor..<cursor + 4])
        cursor += 4
        guard magic == headerMagic else {
            throw ANNSError.corruptFile("Invalid file magic")
        }

        let formatVersion = try readUInt32(payload, &cursor)
        guard formatVersion == version else {
            throw ANNSError.corruptFile("Unsupported file version")
        }

        let nodeCount = Int(try readUInt32(payload, &cursor))
        let degree = Int(try readUInt32(payload, &cursor))
        let dim = Int(try readUInt32(payload, &cursor))
        let metricCode = try readUInt32(payload, &cursor)
        let metric = try metric(from: metricCode)

        let vectorByteCount = nodeCount * dim * MemoryLayout<Float>.stride
        let adjacencyByteCount = nodeCount * degree * MemoryLayout<UInt32>.stride
        let distanceByteCount = nodeCount * degree * MemoryLayout<Float>.stride

        let expectedBodyLength = vectorByteCount + adjacencyByteCount + distanceByteCount
        guard payload.count >= cursor + expectedBodyLength + 4 else {
            throw ANNSError.corruptFile("Corrupt or truncated body")
        }

        let vectorData = payload[cursor..<cursor + vectorByteCount]
        cursor += vectorByteCount

        let adjacencyData = payload[cursor..<cursor + adjacencyByteCount]
        cursor += adjacencyByteCount

        let distanceData = payload[cursor..<cursor + distanceByteCount]
        cursor += distanceByteCount

        let idMapByteCount = Int(try readUInt32(payload, &cursor))
        guard payload.count >= cursor + idMapByteCount + 4 else {
            throw ANNSError.corruptFile("Truncated or invalid IDMap payload")
        }

        let idMapData = payload[cursor..<cursor + idMapByteCount]
        cursor += idMapByteCount

        let entryPoint = try readUInt32(payload, &cursor)

        if cursor > payload.count {
            throw ANNSError.corruptFile("Malformed payload")
        }

        let vectors = try VectorBuffer(capacity: nodeCount, dim: dim, device: device)
        let graph = try GraphBuffer(capacity: nodeCount, degree: degree, device: device)

        vectorData.withUnsafeBytes { raw in
            vectors.buffer.contents().copyMemory(from: raw.baseAddress!, byteCount: vectorByteCount)
        }
        adjacencyData.withUnsafeBytes { raw in
            graph.adjacencyBuffer.contents().copyMemory(from: raw.baseAddress!, byteCount: adjacencyByteCount)
        }
        distanceData.withUnsafeBytes { raw in
            graph.distanceBuffer.contents().copyMemory(from: raw.baseAddress!, byteCount: distanceByteCount)
        }

        vectors.setCount(nodeCount)
        graph.setCount(nodeCount)

        let idMap: IDMap
        do {
            idMap = try JSONDecoder().decode(IDMap.self, from: idMapData)
        } catch {
            throw ANNSError.corruptFile("IDMap payload is corrupt")
        }

        return (vectors: vectors, graph: graph, idMap: idMap, entryPoint: entryPoint, metric: metric)
    }

    private static func metricCode(_ metric: Metric) -> UInt32 {
        switch metric {
        case .cosine:
            return 0
        case .l2:
            return 1
        case .innerProduct:
            return 2
        }
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

    private static func append(uint32 value: UInt32, to payload: inout Data) {
        var littleEndianValue = value.littleEndian
        payload.append(Data(bytes: &littleEndianValue, count: MemoryLayout<UInt32>.size))
    }

    private static func append(magic: [UInt8], to payload: inout Data) {
        payload.append(contentsOf: magic)
    }

    private static func readUInt32(_ payload: Data, _ cursor: inout Int) throws -> UInt32 {
        guard cursor + MemoryLayout<UInt32>.size <= payload.count else {
            throw ANNSError.corruptFile("Unexpected EOF")
        }

        let value: UInt32 = payload.withUnsafeBytes { raw in
            let start = raw.baseAddress!.advanced(by: cursor).assumingMemoryBound(to: UInt8.self)
            return UInt32(start[0]) |
                (UInt32(start[1]) << 8) |
                (UInt32(start[2]) << 16) |
                (UInt32(start[3]) << 24)
        }
        cursor += MemoryLayout<UInt32>.size
        return value
    }
}
