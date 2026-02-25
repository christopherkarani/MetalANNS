import Testing
@testable import MetalANNS
import Darwin

@Suite("Swift 6.2 Modernization Tests")
struct Swift62ModernizationTests {
    @Test("typedThrowCatchesIndexEmpty")
    func typedThrowCatchesIndexEmpty() async {
        let index = ANNSIndex()
        do {
            _ = try await index.search(query: [1.0, 2.0, 3.0], k: 5)
        } catch .indexEmpty {
            // Typed throws allows direct enum pattern matching.
        } catch {
            Issue.record("Expected .indexEmpty, got \(error)")
        }
    }

    @Test("typedThrowCatchesDimensionMismatch")
    func typedThrowCatchesDimensionMismatch() async throws {
        let vectors = (0..<50).map { i in (0..<8).map { d in Float(i * 8 + d) } }
        let ids = (0..<50).map { "v_\($0)" }
        let index = ANNSIndex(configuration: IndexConfiguration(degree: 4, metric: .l2))
        try await index.build(vectors: vectors, ids: ids)

        do {
            _ = try await index.search(query: [1.0], k: 5)
        } catch .dimensionMismatch(let expected, let got) {
            #expect(expected == 8)
            #expect(got == 1)
        } catch {
            Issue.record("Expected .dimensionMismatch, got \(error)")
        }
    }

    @Test("newErrorCasesExist")
    func newErrorCasesExist() {
        let e1: ANNSError = .serializationFailed("test")
        let e2: ANNSError = .metalError("test")
        let e3: ANNSError = .invalidArgument("test")
        let errors: [ANNSError] = [e1, e2, e3]
        #expect(errors.count == 3)
    }

    @Test("concurrentSearchesRunInParallel")
    func concurrentSearchesRunInParallel() async throws {
        let vectors: [[Float]] = (0..<100).map { i in
            (0..<16).map { d in
                let input = Double(Float(i * 16 + d) * 0.173)
                return Float(Darwin.sin(input))
            }
        }
        let ids = (0..<100).map { "v_\($0)" }
        let index = ANNSIndex(configuration: IndexConfiguration(degree: 8, metric: .cosine))
        try await index.build(vectors: vectors, ids: ids)

        try await withThrowingTaskGroup(of: [SearchResult].self) { group in
            for i in 0..<10 {
                group.addTask {
                    try await index.search(query: vectors[i], k: 5)
                }
            }

            var completedCount = 0
            for try await results in group {
                #expect(!results.isEmpty)
                #expect(results.count == 5)
                completedCount += 1
            }
            #expect(completedCount == 10)
        }
    }

    @Test("concurrentCountAccess")
    func concurrentCountAccess() async throws {
        let vectors = (0..<50).map { i in (0..<8).map { d in Float(i * 8 + d) } }
        let ids = (0..<50).map { "v_\($0)" }
        let index = ANNSIndex(configuration: IndexConfiguration(degree: 4, metric: .l2))
        try await index.build(vectors: vectors, ids: ids)

        await withTaskGroup(of: Int.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    await index.count
                }
            }

            for await count in group {
                #expect(count == 50)
            }
        }
    }
}
