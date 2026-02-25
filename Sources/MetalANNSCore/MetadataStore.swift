import Foundation

public struct MetadataStore: Sendable, Codable {
    private var stringColumns: [String: [UInt32: String]] = [:]
    private var floatColumns: [String: [UInt32: Float]] = [:]
    private var intColumns: [String: [UInt32: Int64]] = [:]

    public init() {}

    public mutating func set(_ column: String, stringValue: String, for id: UInt32) {
        stringColumns[column, default: [:]][id] = stringValue
    }

    public mutating func set(_ column: String, floatValue: Float, for id: UInt32) {
        floatColumns[column, default: [:]][id] = floatValue
    }

    public mutating func set(_ column: String, intValue: Int64, for id: UInt32) {
        intColumns[column, default: [:]][id] = intValue
    }

    public func getString(_ column: String, for id: UInt32) -> String? {
        stringColumns[column]?[id]
    }

    public func getFloat(_ column: String, for id: UInt32) -> Float? {
        floatColumns[column]?[id]
    }

    public func getInt(_ column: String, for id: UInt32) -> Int64? {
        intColumns[column]?[id]
    }

    public func matches(id: UInt32, filter: SearchFilter) -> Bool {
        switch filter {
        case .equals(let column, let value):
            return stringColumns[column]?[id] == value
        case .greaterThan(let column, let value):
            if let floatValue = floatColumns[column]?[id] {
                return floatValue > value
            }
            if let intValue = intColumns[column]?[id] {
                return Float(intValue) > value
            }
            return false
        case .lessThan(let column, let value):
            if let floatValue = floatColumns[column]?[id] {
                return floatValue < value
            }
            if let intValue = intColumns[column]?[id] {
                return Float(intValue) < value
            }
            return false
        case .in(let column, let values):
            guard let value = stringColumns[column]?[id] else {
                return false
            }
            return values.contains(value)
        case .and(let filters):
            return filters.allSatisfy { matches(id: id, filter: $0) }
        case .or(let filters):
            return filters.contains { matches(id: id, filter: $0) }
        case .not(let inner):
            return !matches(id: id, filter: inner)
        }
    }

    public func remapped(using mapping: [UInt32: UInt32]) -> MetadataStore {
        var result = MetadataStore()

        for (column, values) in stringColumns {
            for (oldID, value) in values {
                if let newID = mapping[oldID] {
                    result.stringColumns[column, default: [:]][newID] = value
                }
            }
        }

        for (column, values) in floatColumns {
            for (oldID, value) in values {
                if let newID = mapping[oldID] {
                    result.floatColumns[column, default: [:]][newID] = value
                }
            }
        }

        for (column, values) in intColumns {
            for (oldID, value) in values {
                if let newID = mapping[oldID] {
                    result.intColumns[column, default: [:]][newID] = value
                }
            }
        }

        return result
    }

    public mutating func remove(id: UInt32) {
        for column in Array(stringColumns.keys) {
            stringColumns[column]?[id] = nil
            if stringColumns[column]?.isEmpty == true {
                stringColumns[column] = nil
            }
        }
        for column in Array(floatColumns.keys) {
            floatColumns[column]?[id] = nil
            if floatColumns[column]?.isEmpty == true {
                floatColumns[column] = nil
            }
        }
        for column in Array(intColumns.keys) {
            intColumns[column]?[id] = nil
            if intColumns[column]?.isEmpty == true {
                intColumns[column] = nil
            }
        }
    }

    public var isEmpty: Bool {
        stringColumns.isEmpty && floatColumns.isEmpty && intColumns.isEmpty
    }
}
