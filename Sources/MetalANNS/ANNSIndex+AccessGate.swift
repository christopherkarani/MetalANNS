import Foundation

/// Read/write gate over the index's shared GPU buffers.
///
/// Actor isolation alone is not enough: searches suspend while GPU or host
/// scans read `vectors` and `graph` (reference types shared with Metal), and
/// `build`/`compact` suspend between reading state and installing the result.
/// Without the gate, an insert could mutate a buffer mid-scan, or land during
/// `compact()` and be discarded when the compacted state is installed.
///
/// - Readers (`search`, `batchSearch`, `rangeSearch`) run concurrently.
/// - Writers (`build`, `insert`, `batchInsert`, `compact`, `repair`) are
///   exclusive: they wait for in-flight readers to finish, and hold the gate
///   across their own suspension points.
/// - Writers have priority: once one is waiting, new readers queue behind it.
///
/// `delete` and `setMetadata` only touch value-type state that each search
/// snapshots up front, so they do not need the gate.
///
/// Never acquire the gate re-entrantly (e.g. `batchSearch` calling the public
/// `search`): a queued writer would deadlock against the outer reader.
extension GraphIndex {
    internal func beginRead() async {
        while writerActive || waitingWriters > 0 {
            await withCheckedContinuation { gateWaiters.append($0) }
        }
        activeReaders += 1
    }

    internal func endRead() {
        activeReaders -= 1
        if activeReaders == 0 {
            wakeGateWaiters()
        }
    }

    internal func beginWrite() async {
        waitingWriters += 1
        while writerActive || activeReaders > 0 {
            await withCheckedContinuation { gateWaiters.append($0) }
        }
        waitingWriters -= 1
        writerActive = true
    }

    internal func endWrite() {
        writerActive = false
        cpuGraphCache = nil
        wakeGateWaiters()
    }

    /// Wakes every waiter; each re-checks its condition on resumption. Waiter
    /// counts are tiny (in-flight API calls), so this is simpler than a FIFO
    /// handoff and cannot lose a wake-up.
    private func wakeGateWaiters() {
        guard !gateWaiters.isEmpty else { return }
        let waiters = gateWaiters
        gateWaiters.removeAll(keepingCapacity: true)
        for waiter in waiters {
            waiter.resume()
        }
    }
}
