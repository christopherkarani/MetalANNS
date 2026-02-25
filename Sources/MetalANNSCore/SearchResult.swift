public struct SearchResult: Sendable {
    public let id: String
    public let score: Float
    public let internalID: UInt32

    public init(id: String, score: Float, internalID: UInt32) {
        self.id = id
        self.score = score
        self.internalID = internalID
    }
}
