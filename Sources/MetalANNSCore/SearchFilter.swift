import Foundation

public indirect enum _LegacySearchFilter: Sendable {
    case equals(column: String, value: String)
    case greaterThan(column: String, value: Float)
    case lessThan(column: String, value: Float)
    case greaterThanInt(column: String, value: Int64)
    case lessThanInt(column: String, value: Int64)
    case `in`(column: String, values: Set<String>)
    case and([_LegacySearchFilter])
    case or([_LegacySearchFilter])
    case not(_LegacySearchFilter)
}
