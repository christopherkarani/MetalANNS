import Foundation

public struct ProductQuantizer: Sendable, Codable {
    public let numSubspaces: Int
    public let centroidsPerSubspace: Int
    public let subspaceDimension: Int
    public let codebooks: [[[Float]]]

    public static func train(
        vectors: [[Float]],
        numSubspaces: Int = 8,
        centroidsPerSubspace: Int = 256,
        maxIterations: Int = 20
    ) throws -> ProductQuantizer {
        guard !vectors.isEmpty else {
            throw ANNSError.constructionFailed("Cannot train ProductQuantizer with empty vectors")
        }
        guard numSubspaces > 0 else {
            throw ANNSError.constructionFailed("numSubspaces must be greater than zero")
        }
        guard centroidsPerSubspace == 256 else {
            throw ANNSError.constructionFailed("ProductQuantizer requires centroidsPerSubspace == 256")
        }

        let dimension = vectors[0].count
        guard dimension > 0 else {
            throw ANNSError.dimensionMismatch(expected: 1, got: 0)
        }
        guard vectors.count >= centroidsPerSubspace else {
            throw ANNSError.constructionFailed(
                "Need at least \(centroidsPerSubspace) vectors to train ProductQuantizer"
            )
        }

        for vector in vectors where vector.count != dimension {
            throw ANNSError.dimensionMismatch(expected: dimension, got: vector.count)
        }
        guard dimension.isMultiple(of: numSubspaces) else {
            throw ANNSError.constructionFailed(
                "Vector dimension \(dimension) must be divisible by numSubspaces \(numSubspaces)"
            )
        }

        let subspaceDimension = dimension / numSubspaces
        var codebooks: [[[Float]]] = []
        codebooks.reserveCapacity(numSubspaces)

        for subspace in 0..<numSubspaces {
            let start = subspace * subspaceDimension
            let end = start + subspaceDimension
            let subVectors = vectors.map { Array($0[start..<end]) }

            let result = try KMeans.cluster(
                vectors: subVectors,
                k: centroidsPerSubspace,
                maxIterations: maxIterations,
                metric: .l2
            )
            codebooks.append(result.centroids)
        }

        return ProductQuantizer(
            numSubspaces: numSubspaces,
            centroidsPerSubspace: centroidsPerSubspace,
            subspaceDimension: subspaceDimension,
            codebooks: codebooks
        )
    }

    public func encode(vector: [Float]) throws -> [UInt8] {
        let expectedDimension = numSubspaces * subspaceDimension
        guard vector.count == expectedDimension else {
            throw ANNSError.dimensionMismatch(expected: expectedDimension, got: vector.count)
        }

        var encoded = [UInt8](repeating: 0, count: numSubspaces)
        for subspace in 0..<numSubspaces {
            let start = subspace * subspaceDimension
            let end = start + subspaceDimension
            let subVector = Array(vector[start..<end])
            let centroids = codebooks[subspace]

            var bestIndex = 0
            var bestDistance = Float.greatestFiniteMagnitude
            for centroidIndex in 0..<centroidsPerSubspace {
                let distance = SIMDDistance.distance(subVector, centroids[centroidIndex], metric: .l2)
                if distance < bestDistance {
                    bestDistance = distance
                    bestIndex = centroidIndex
                }
            }
            encoded[subspace] = UInt8(bestIndex)
        }
        return encoded
    }

    public func reconstruct(codes: [UInt8]) throws -> [Float] {
        guard codes.count == numSubspaces else {
            throw ANNSError.dimensionMismatch(expected: numSubspaces, got: codes.count)
        }

        var vector: [Float] = []
        vector.reserveCapacity(numSubspaces * subspaceDimension)

        for subspace in 0..<numSubspaces {
            let centroidIndex = Int(codes[subspace])
            guard centroidIndex < centroidsPerSubspace else {
                throw ANNSError.constructionFailed("Code \(centroidIndex) out of range")
            }
            vector.append(contentsOf: codebooks[subspace][centroidIndex])
        }
        return vector
    }

    public func approximateDistance(query: [Float], codes: [UInt8], metric: Metric) -> Float {
        guard let table = distanceTable(query: query, metric: metric), codes.count == numSubspaces else {
            return Float.greatestFiniteMagnitude
        }

        var distance: Float = 0
        for subspace in 0..<numSubspaces {
            let code = Int(codes[subspace])
            if code >= centroidsPerSubspace {
                return Float.greatestFiniteMagnitude
            }
            distance += table[subspace][code]
        }
        return distance
    }

    func distanceTable(query: [Float], metric: Metric) -> [[Float]]? {
        let expectedDimension = numSubspaces * subspaceDimension
        guard query.count == expectedDimension else {
            return nil
        }

        var table = [[Float]](
            repeating: [Float](repeating: 0, count: centroidsPerSubspace),
            count: numSubspaces
        )

        for subspace in 0..<numSubspaces {
            let start = subspace * subspaceDimension
            let end = start + subspaceDimension
            let subQuery = Array(query[start..<end])
            let centroids = codebooks[subspace]

            for centroidIndex in 0..<centroidsPerSubspace {
                table[subspace][centroidIndex] = SIMDDistance.distance(
                    subQuery,
                    centroids[centroidIndex],
                    metric: metric
                )
            }
        }

        return table
    }
}
