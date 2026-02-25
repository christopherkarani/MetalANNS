import Foundation

public indirect enum SearchFilter: Sendable {
    case equals(column: String, value: String)
    case greaterThan(column: String, value: Float)
    case lessThan(column: String, value: Float)
    case `in`(column: String, values: Set<String>)
    case and([SearchFilter])
    case or([SearchFilter])
    case not(SearchFilter)
}
