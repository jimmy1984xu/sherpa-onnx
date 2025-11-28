package com.k2fsa.sherpa.onnx

/**
 * Fast Clustering for speaker clustering.
 *
 * This class provides a standalone Fast Clustering implementation that can be used
 * independently without requiring speaker segmentation or embedding extraction models.
 *
 * Usage example:
 * ```
 * val config = FastClusteringConfig(
 *     numClusters = -1,  // Use threshold mode
 *     threshold = 0.5f
 * )
 * val clustering = FastClustering(config)
 *
 * // embeddings: FloatArray of shape (num_segments * embedding_dim) in row-major order
 * // numSegments: number of segments
 * // embeddingDim: dimension of each embedding vector
 * val labels = clustering.cluster(embeddings, numSegments, embeddingDim)
 * ```
 */
class FastClustering(val config: FastClusteringConfig) {
    private var ptr: Long

    init {
        ptr = newFromConfig(config)
        if (ptr == 0L) {
            throw RuntimeException("Failed to create FastClustering instance")
        }
    }

    protected fun finalize() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0
        }
    }

    fun release() = finalize()

    /**
     * Perform clustering on embeddings.
     *
     * @param embeddings FloatArray containing embeddings in row-major order.
     *                   Shape: (numSegments * embeddingDim)
     * @param numSegments Number of segments (rows)
     * @param embeddingDim Dimension of each embedding vector (columns)
     * @return IntArray of cluster labels, one for each segment
     */
    fun cluster(
        embeddings: FloatArray,
        numSegments: Int,
        embeddingDim: Int,
    ): IntArray {
        if (embeddings.size != numSegments * embeddingDim) {
            throw IllegalArgumentException(
                "Embeddings size (${embeddings.size}) does not match " +
                    "numSegments * embeddingDim ($numSegments * $embeddingDim = ${numSegments * embeddingDim})"
            )
        }
        return cluster(ptr, embeddings, numSegments, embeddingDim)
    }

    private external fun newFromConfig(config: FastClusteringConfig): Long

    private external fun delete(ptr: Long)

    private external fun cluster(
        ptr: Long,
        embeddings: FloatArray,
        numSegments: Int,
        embeddingDim: Int,
    ): IntArray

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}

