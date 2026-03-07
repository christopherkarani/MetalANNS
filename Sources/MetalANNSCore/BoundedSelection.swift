import Foundation

struct BinaryHeap<Element> {
    private var storage: [Element] = []
    private let areSorted: (Element, Element) -> Bool

    init(sort: @escaping (Element, Element) -> Bool) {
        self.areSorted = sort
    }

    var count: Int { storage.count }
    var isEmpty: Bool { storage.isEmpty }
    var peek: Element? { storage.first }

    mutating func push(_ element: Element) {
        storage.append(element)
        siftUp(from: storage.count - 1)
    }

    @discardableResult
    mutating func pop() -> Element? {
        guard !storage.isEmpty else {
            return nil
        }
        if storage.count == 1 {
            return storage.removeLast()
        }

        let first = storage[0]
        storage[0] = storage.removeLast()
        siftDown(from: 0)
        return first
    }

    mutating func replaceTop(with element: Element) {
        guard !storage.isEmpty else {
            push(element)
            return
        }
        storage[0] = element
        siftDown(from: 0)
    }

    func unorderedElements() -> [Element] {
        storage
    }

    private mutating func siftUp(from index: Int) {
        var child = index
        var parent = (child - 1) / 2

        while child > 0 && areSorted(storage[child], storage[parent]) {
            storage.swapAt(child, parent)
            child = parent
            parent = (child - 1) / 2
        }
    }

    private mutating func siftDown(from index: Int) {
        var parent = index

        while true {
            let left = (2 * parent) + 1
            let right = left + 1
            var candidate = parent

            if left < storage.count && areSorted(storage[left], storage[candidate]) {
                candidate = left
            }
            if right < storage.count && areSorted(storage[right], storage[candidate]) {
                candidate = right
            }
            if candidate == parent {
                return
            }

            storage.swapAt(parent, candidate)
            parent = candidate
        }
    }
}

struct BoundedPriorityBuffer<Element> {
    private(set) var heap: BinaryHeap<Element>
    let capacity: Int
    private let outranks: (Element, Element) -> Bool

    init(capacity: Int, outranks: @escaping (Element, Element) -> Bool) {
        self.capacity = max(0, capacity)
        self.outranks = outranks
        // Keep the worst element at the root so replacement is O(log k).
        self.heap = BinaryHeap(sort: { lhs, rhs in outranks(rhs, lhs) })
    }

    var count: Int { heap.count }
    var worst: Element? { heap.peek }

    mutating func insert(_ element: Element) {
        guard capacity > 0 else {
            return
        }
        if heap.count < capacity {
            heap.push(element)
            return
        }
        guard let worst else {
            heap.push(element)
            return
        }
        guard outranks(element, worst) else {
            return
        }
        heap.replaceTop(with: element)
    }

    func sortedElements() -> [Element] {
        heap.unorderedElements().sorted(by: outranks)
    }
}
