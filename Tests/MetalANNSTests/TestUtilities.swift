import Foundation

/// Deterministic XOR-shift PRNG for reproducible test data.
struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64

    init(state: UInt64) {
        self.state = state == 0 ? 1 : state
    }

    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}
