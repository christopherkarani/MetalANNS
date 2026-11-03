/// Distance metric. Every search API ranks by a *distance* where lower is
/// closer; `similarity(fromDistance:)` converts it to a score where higher is
/// closer.
///
/// | Metric | distance | similarity |
/// |---|---|---|
/// | `cosine` | `1 − cos θ`, in `[0, 2]` | `cos θ`, in `[−1, 1]` |
/// | `innerProduct` | `−(a · b)` | `a · b` |
/// | `l2` | squared Euclidean `‖a − b‖²` | `−‖a − b‖²` |
/// | `hamming` | number of differing bits | `−(differing bits)` |
///
/// Cosine is computed on the stored vectors as given (no normalization at
/// insert time), so it is exact for vectors of any non-zero norm.
public enum Metric: String, Sendable, Codable {
    case cosine
    case l2
    case innerProduct
    case hamming

    /// Converts a distance produced by this metric into a similarity where
    /// higher is closer. Ordering by descending similarity equals ordering by
    /// ascending distance.
    @inlinable
    public func similarity(fromDistance distance: Float) -> Float {
        switch self {
        case .cosine:
            1 - distance
        case .l2, .innerProduct, .hamming:
            -distance
        }
    }
}
