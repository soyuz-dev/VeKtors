package org.soyuz.vektors

class Tensor private constructor(
    private val data: FloatArray,
    shape: List<Int>,
) : TensorLike {

    override val shape: List<Int> =
        shape.toList()

    override fun getFlat(index: Int): Float {
        require(index in data.indices) {
            "Flat index out of bounds: $index"
        }

        return data[index]
    }

    override operator fun get(vararg indices: Int): Float =
        data[flatIndex(indices)]

    private fun flatIndex(indices: IntArray): Int {
        require(indices.size == rank) {
            "Expected $rank indices, got ${indices.size}"
        }

        var index = 0
        var stride = 1

        for (dimension in rank - 1 downTo 0) {
            require(indices[dimension] in 0 until shape[dimension]) {
                "Index ${indices[dimension]} out of bounds " +
                        "for dimension $dimension with size ${shape[dimension]}"
            }

            index += indices[dimension] * stride
            stride *= shape[dimension]
        }

        return index
    }

    companion object {
        operator fun invoke(
            shape: List<Int>,
            init: (IntArray) -> Float,
        ): Tensor {
            requireShape(shape)

            val data = FloatArray(sizeOf(shape))

            for (flatIndex in data.indices) {
                data[flatIndex] =
                    init(indicesOf(flatIndex, shape))
            }

            return Tensor(data, shape)
        }

        operator fun invoke(
            vararg shape: Int,
            init: (IntArray) -> Float,
        ): Tensor =
            invoke(shape.asList(), init)

        operator fun invoke(other: TensorLike): Tensor =
            fromFlat(
                other.shape,
                FloatArray(other.size) {
                    other.getFlat(it)
                }
            )

        internal fun fromFlat(
            shape: List<Int>,
            data: FloatArray,
        ): Tensor {
            requireShape(shape)

            val expectedSize = sizeOf(shape)

            require(data.size == expectedSize) {
                "Expected $expectedSize values, got ${data.size}"
            }

            return Tensor(
                data.copyOf(),
                shape,
            )
        }

        private fun requireShape(shape: List<Int>) {
            require(shape.isNotEmpty()) {
                "Tensor must have at least one dimension"
            }

            require(shape.all { it > 0 }) {
                "Tensor dimensions must be positive"
            }
        }

        private fun sizeOf(shape: List<Int>): Int =
            shape.fold(1) { acc, dimension ->
                acc * dimension
            }

        private fun indicesOf(
            flatIndex: Int,
            shape: List<Int>,
        ): IntArray {
            var remainder = flatIndex
            val indices = IntArray(shape.size)

            for (dimension in shape.size - 1 downTo 0) {
                indices[dimension] =
                    remainder % shape[dimension]

                remainder /= shape[dimension]
            }

            return indices
        }
    }
}