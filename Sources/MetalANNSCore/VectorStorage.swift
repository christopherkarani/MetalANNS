import Foundation
import Metal

/// Abstraction over Float32 and Float16 vector storage.
/// Both buffer types conform so construction, search, and pruning can work generically.
public protocol VectorStorage: AnyObject, Sendable {
    var buffer: MTLBuffer { get }
    var dim: Int { get }
    var capacity: Int { get }
    var count: Int { get }
    var isFloat16: Bool { get }

    func setCount(_ newCount: Int)
    func insert(vector: [Float], at index: Int) throws(ANNSError)
    func batchInsert(vectors: [[Float]], startingAt start: Int) throws(ANNSError)
    func vector(at index: Int) -> [Float]
}
