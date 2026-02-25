import Accelerate

public enum SIMDDistance {
    public static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        guard !a.isEmpty else {
            return 1.0
        }

        return a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                cosine(aBuffer.baseAddress!, bBuffer.baseAddress!, dim: a.count)
            }
        }
    }

    public static func cosine(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, dim: Int) -> Float {
        precondition(dim >= 0)
        guard dim > 0 else {
            return 1.0
        }

        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(dim))
        vDSP_dotpr(a, 1, a, 1, &normA, vDSP_Length(dim))
        vDSP_dotpr(b, 1, b, 1, &normB, vDSP_Length(dim))

        let denom = sqrt(normA) * sqrt(normB)
        return denom < 1e-10 ? 1.0 : (1.0 - (dot / denom))
    }

    public static func l2(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        guard !a.isEmpty else {
            return 0
        }

        return a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                l2(aBuffer.baseAddress!, bBuffer.baseAddress!, dim: a.count)
            }
        }
    }

    public static func l2(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, dim: Int) -> Float {
        precondition(dim >= 0)
        guard dim > 0 else {
            return 0
        }

        var distance: Float = 0
        vDSP_distancesq(a, 1, b, 1, &distance, vDSP_Length(dim))
        return distance
    }

    public static func innerProduct(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        guard !a.isEmpty else {
            return 0
        }

        return a.withUnsafeBufferPointer { aBuffer in
            b.withUnsafeBufferPointer { bBuffer in
                innerProduct(aBuffer.baseAddress!, bBuffer.baseAddress!, dim: a.count)
            }
        }
    }

    public static func innerProduct(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, dim: Int) -> Float {
        precondition(dim >= 0)
        guard dim > 0 else {
            return 0
        }

        var dot: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(dim))
        return -dot
    }

    public static func distance(_ a: [Float], _ b: [Float], metric: Metric) -> Float {
        switch metric {
        case .cosine:
            cosine(a, b)
        case .l2:
            l2(a, b)
        case .innerProduct:
            innerProduct(a, b)
        }
    }

    public static func distance(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        dim: Int,
        metric: Metric
    ) -> Float {
        switch metric {
        case .cosine:
            cosine(a, b, dim: dim)
        case .l2:
            l2(a, b, dim: dim)
        case .innerProduct:
            innerProduct(a, b, dim: dim)
        }
    }
}
