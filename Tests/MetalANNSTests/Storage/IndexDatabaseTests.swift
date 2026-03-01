import Testing
import Foundation
import GRDB
@testable import MetalANNSCore
@testable import MetalANNS

@Suite("IndexDatabase Tests")
struct IndexDatabaseTests {
    private func tempDBPath() -> String {
        let fileName = "index-" + UUID().uuidString + ".db"
        return (NSTemporaryDirectory() as NSString).appendingPathComponent(fileName)
    }

    @Test("Create and open database")
    func createAndOpenDatabase() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let db = try IndexDatabase(path: path)
        #expect(FileManager.default.fileExists(atPath: path))

        let tables = try db.pool.read { db in
            try String.fetchAll(db, sql: """
                SELECT name FROM sqlite_master
                WHERE type='table'
                  AND name NOT LIKE 'sqlite_%'
                  AND name != 'grdb_migrations'
                ORDER BY name
                """)
        }
        #expect(tables == ["config", "idmap", "soft_deletion"])
    }

    @Test("Reopen existing database")
    func reopenExistingDatabase() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        _ = try IndexDatabase(path: path)
        let db = try IndexDatabase(path: path)
        let count = try db.pool.read { db in
            try Int.fetchOne(db, sql: "SELECT COUNT(*) FROM idmap") ?? 0
        }
        #expect(count == 0)
    }

    @Test("Save and load idMap")
    func saveAndLoadIDMap() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let db = try IndexDatabase(path: path)
        var idMap = IDMap()
        _ = idMap.assign(externalID: "alpha")
        _ = idMap.assign(externalID: "beta")
        _ = idMap.assign(externalID: "gamma")

        try db.saveIDMap(idMap)
        let loaded = try db.loadIDMap()

        #expect(loaded.count == 3)
        #expect(loaded.internalID(for: "alpha") == 0)
        #expect(loaded.internalID(for: "beta") == 1)
        #expect(loaded.internalID(for: "gamma") == 2)
        #expect(loaded.externalID(for: 0) == "alpha")
        #expect(loaded.externalID(for: 1) == "beta")
        #expect(loaded.externalID(for: 2) == "gamma")
    }

    @Test("Saving idMap overwrites previous values")
    func saveIDMapOverwritesPrevious() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let db = try IndexDatabase(path: path)
        var original = IDMap()
        _ = original.assign(externalID: "old")
        try db.saveIDMap(original)

        var replacement = IDMap()
        _ = replacement.assign(externalID: "new-a")
        _ = replacement.assign(externalID: "new-b")
        try db.saveIDMap(replacement)

        let loaded = try db.loadIDMap()
        #expect(loaded.count == 2)
        #expect(loaded.internalID(for: "old") == nil)
        #expect(loaded.internalID(for: "new-a") == 0)
        #expect(loaded.internalID(for: "new-b") == 1)
    }

    @Test("Load idMap from empty database")
    func loadIDMapFromEmptyDatabase() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let db = try IndexDatabase(path: path)
        let loaded = try db.loadIDMap()

        #expect(loaded.count == 0)
    }

    @Test("Save and load configuration")
    func saveAndLoadConfig() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let db = try IndexDatabase(path: path)
        var configuration = IndexConfiguration.default
        configuration.metric = .l2
        configuration.useFloat16 = true
        configuration.degree = 32

        try db.saveConfiguration(configuration)
        let loaded = try db.loadConfiguration()

        #expect(loaded?.metric == .l2)
        #expect(loaded?.useFloat16 == true)
        #expect(loaded?.degree == 32)
    }

    @Test("Save and load soft-deletion")
    func saveAndLoadSoftDeletion() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let db = try IndexDatabase(path: path)
        var softDeletion = SoftDeletion()
        softDeletion.markDeleted(3)
        softDeletion.markDeleted(7)
        softDeletion.markDeleted(15)

        try db.saveSoftDeletion(softDeletion)
        let loaded = try db.loadSoftDeletion()

        #expect(loaded.deletedCount == 3)
        #expect(loaded.isDeleted(3))
        #expect(loaded.isDeleted(7))
        #expect(loaded.isDeleted(15))
        #expect(!loaded.isDeleted(0))
    }

    @Test("Save and load metadata store")
    func saveAndLoadMetadataStore() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let db = try IndexDatabase(path: path)
        var store = MetadataStore()
        store.set("color", stringValue: "red", for: 0)
        store.set("score", floatValue: 0.95, for: 0)
        store.set("color", stringValue: "blue", for: 1)
        store.set("priority", intValue: 42, for: 1)

        try db.saveMetadataStore(store)
        let loaded = try db.loadMetadataStore()

        #expect(loaded.getString("color", for: 0) == "red")
        #expect(loaded.getString("color", for: 1) == "blue")
        #expect(loaded.getFloat("score", for: 0) == 0.95)
        #expect(loaded.getInt("priority", for: 1) == 42)
    }

    @Test("Load empty metadata store")
    func loadEmptyMetadataStore() throws {
        let path = tempDBPath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let db = try IndexDatabase(path: path)
        let loaded = try db.loadMetadataStore()

        #expect(loaded.isEmpty)
    }
}
