public struct SearchResult: Sendable {
    public let id: String
    public let score: Float
    public let internalID: UInt32
    public let numericID: UInt64?

    public init(id: String, score: Float, internalID: UInt32, numericID: UInt64? = nil) {
        self.id = id
        self.score = score
        self.internalID = internalID
        self.numericID = numericID
    }
}
