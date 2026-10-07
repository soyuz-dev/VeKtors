package org.soyuz.vektors

class MutableTensor private constructor(
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

    operator fun set(
        vararg indices: Int,
        value: Float,
    ) {
        data[flatIndex(indices)] = value
    }

    fun setFlat(index: Int, value: Float) {
        require(index in data.indices) {
            "Flat index out of bounds: $index"
        }

        data[index] = value
    }

    operator fun plusAssign(other: TensorLike) {
        requireMatchShape(other)

        for (i in data.indices) {
            data[i] += other.getFlat(i)
        }
    }

    operator fun minusAssign(other: TensorLike) {
        requireMatchShape(other)

        for (i in data.indices) {
            data[i] -= other.getFlat(i)
        }
    }

    operator fun timesAssign(other: Float) {
        for (i in data.indices) {
            data[i] *= other
        }
    }

    operator fun divAssign(other: Float) {
        for (i in data.indices) {
            data[i] /= other
        }
    }

    fun mapInPlace(transform: (Float) -> Float) {
        for (i in data.indices) {
            data[i] = transform(data[i])
        }
    }

    fun toTensor(): Tensor =
        Tensor(this)

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

    private fun requireMatchShape(other: TensorLike) =
        require(shape == other.shape) {
            "Tensor shapes must match: $shape != ${other.shape}"
        }

    companion object {
        operator fun invoke(
            shape: List<Int>,
            init: (IntArray) -> Float,
        ): MutableTensor {
            requireShape(shape)

            val data = FloatArray(sizeOf(shape))

            for (flatIndex in data.indices) {
                data[flatIndex] =
                    init(indicesOf(flatIndex, shape))
            }

            return MutableTensor(data, shape)
        }

        operator fun invoke(
            vararg shape: Int,
            init: (IntArray) -> Float,
        ): MutableTensor =
            invoke(shape.asList(), init)

        operator fun invoke(
            other: TensorLike,
        ): MutableTensor =
            MutableTensor(
                FloatArray(other.size) {
                    other.getFlat(it)
                },
                other.shape,
            )

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