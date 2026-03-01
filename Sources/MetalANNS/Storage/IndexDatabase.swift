import Foundation
import GRDB
import MetalANNSCore

/// SQLite-backed persistence layer for index structured data.
public final class IndexDatabase: @unchecked Sendable {
    public let pool: DatabasePool

    public init(path: String) throws {
        var configuration = Configuration()
        configuration.prepareDatabase { db in
            try db.execute(sql: "PRAGMA journal_mode = WAL")
            try db.execute(sql: "PRAGMA mmap_size = 268435456")
        }
        self.pool = try DatabasePool(path: path, configuration: configuration)
        try migrate()
    }

    public func prepareForFileMove() throws {
        try pool.writeWithoutTransaction { db in
            try db.execute(sql: "PRAGMA wal_checkpoint(TRUNCATE)")
        }
    }

    private func migrate() throws {
        var migrator = DatabaseMigrator()
        migrator.registerMigration("v1-foundation") { db in
            try db.create(table: "idmap") { table in
                table.column("externalID", .text).notNull().primaryKey()
                table.column("internalID", .integer).notNull().unique()
            }
            try db.create(index: "idmap_by_internal", on: "idmap", columns: ["internalID"])

            try db.create(table: "config") { table in
                table.column("key", .text).notNull().primaryKey()
                table.column("value", .text).notNull()
            }

            try db.create(table: "soft_deletion") { table in
                table.column("internalID", .integer).notNull().primaryKey()
            }
        }
        try migrator.migrate(pool)
    }
}

// MARK: - IDMap persistence

public extension IndexDatabase {
    func saveIDMap(_ idMap: IDMap) throws {
        let rows = (0..<idMap.nextInternalID).compactMap { index -> (String, UInt32)? in
            guard let externalID = idMap.externalID(for: UInt32(index)) else {
                return nil
            }
            return (externalID, UInt32(index))
        }

        try pool.write { db in
            try db.execute(sql: "DELETE FROM idmap")
            let statement = try db.makeStatement(sql: "INSERT INTO idmap (externalID, internalID) VALUES (?, ?)")

            for (externalID, internalID) in rows {
                try statement.execute(arguments: [externalID, Int64(internalID)])
            }

            try db.execute(
                sql: "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                arguments: ["idmap.nextID", String(idMap.nextInternalID)]
            )
        }
    }

    func loadIDMap() throws -> IDMap {
        try pool.read { db in
            let rows = try Row.fetchAll(db, sql: "SELECT externalID, internalID FROM idmap ORDER BY internalID")
            let entries: [(String, UInt32)] = rows.map {
                let externalID: String = $0["externalID"]
                let internalID: Int64 = $0["internalID"]
                return (externalID, UInt32(internalID))
            }

            let nextID = try String.fetchOne(
                db,
                sql: "SELECT value FROM config WHERE key = ?",
                arguments: ["idmap.nextID"]
            ).flatMap(UInt32.init) ?? UInt32(entries.count)

            return IDMap.makeForPersistence(rows: entries, nextID: nextID)
        }
    }
}

// MARK: - Configuration persistence

public extension IndexDatabase {
    func saveConfiguration(_ configuration: IndexConfiguration) throws {
        let data = try JSONEncoder().encode(configuration)
        guard let encoded = String(data: data, encoding: .utf8) else {
            throw ANNSError.constructionFailed("Failed to encode configuration")
        }

        try pool.write { db in
            try db.execute(
                sql: "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                arguments: ["index.configuration", encoded]
            )
        }
    }

    func loadConfiguration() throws -> IndexConfiguration? {
        try pool.read { db in
            guard let payload = try String.fetchOne(
                db,
                sql: "SELECT value FROM config WHERE key = ?",
                arguments: ["index.configuration"]
            ), let data = payload.data(using: .utf8) else {
                return nil
            }
            return try JSONDecoder().decode(IndexConfiguration.self, from: data)
        }
    }
}

// MARK: - Soft-deletion persistence

public extension IndexDatabase {
    func saveSoftDeletion(_ softDeletion: SoftDeletion) throws {
        try pool.write { db in
            try db.execute(sql: "DELETE FROM soft_deletion")

            for internalID in softDeletion.allDeletedIDs {
                try db.execute(
                    sql: "INSERT INTO soft_deletion (internalID) VALUES (?)",
                    arguments: [Int64(internalID)]
                )
            }
        }
    }

    func loadSoftDeletion() throws -> SoftDeletion {
        try pool.read { db in
            let values = try Int64.fetchAll(db, sql: "SELECT internalID FROM soft_deletion")
            var softDeletion = SoftDeletion()
            values.forEach { softDeletion.markDeleted(UInt32($0)) }
            return softDeletion
        }
    }
}

// MARK: - Metadata persistence

public extension IndexDatabase {
    func saveMetadataStore(_ metadataStore: MetadataStore) throws {
        let data = try JSONEncoder().encode(metadataStore)
        guard let encoded = String(data: data, encoding: .utf8) else {
            throw ANNSError.constructionFailed("Failed to encode metadataStore")
        }

        try pool.write { db in
            try db.execute(
                sql: "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                arguments: ["index.metadataStore", encoded]
            )
        }
    }

    func loadMetadataStore() throws -> MetadataStore {
        try pool.read { db in
            guard let payload = try String.fetchOne(
                db,
                sql: "SELECT value FROM config WHERE key = ?",
                arguments: ["index.metadataStore"]
            ), let data = payload.data(using: .utf8) else {
                return MetadataStore()
            }
            return try JSONDecoder().decode(MetadataStore.self, from: data)
        }
    }
}
