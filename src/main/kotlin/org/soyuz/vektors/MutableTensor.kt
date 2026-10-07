package org.soyuz.vektors

class MutableTensor private constructor(
    private val data: FloatArray,
    shape: List<Int>,
    private val strides: List<Int>,
    private val offset: Int,
) : TensorLike {

    override val shape: List<Int> =
        shape.toList()

    override operator fun get(index: Int): MutableTensor {
        require(rank > 0) {
            "Cannot select from a scalar tensor"
        }

        require(index in 0 until shape[0]) {
            "Index $index is out of bounds for dimension 0 with size ${shape[0]}"
        }

        return MutableTensor(
            data = data,
            shape = shape.drop(1),
            strides = strides.drop(1),
            offset = offset + index * strides[0],
        )
    }

    override fun getFlat(index: Int): Float {
        require(index in 0 until size) {
            "Flat index out of bounds: $index"
        }

        return data[dataIndex(index)]
    }

    fun setValueAt(
        vararg indices: Int,
        value: Float,
    ) {
        data[dataIndex(indices)] = value
    }

    fun setFlat(index: Int, value: Float) {
        require(index in 0 until size) {
            "Flat index out of bounds: $index"
        }

        data[dataIndex(index)] = value
    }

    operator fun plusAssign(other: TensorLike) {
        requireMatchShape(other)

        for (i in 0 until size) {
            setFlat(
                i,
                getFlat(i) + other.getFlat(i),
            )
        }
    }

    operator fun minusAssign(other: TensorLike) {
        requireMatchShape(other)

        for (i in 0 until size) {
            setFlat(
                i,
                getFlat(i) - other.getFlat(i),
            )
        }
    }

    operator fun timesAssign(other: Float) {
        for (i in 0 until size) {
            setFlat(
                i,
                getFlat(i) * other,
            )
        }
    }

    operator fun divAssign(other: Float) {
        for (i in 0 until size) {
            setFlat(
                i,
                getFlat(i) / other,
            )
        }
    }

    fun mapInPlace(transform: (Float) -> Float) {
        for (i in 0 until size) {
            setFlat(
                i,
                transform(getFlat(i)),
            )
        }
    }

    fun toTensor(): Tensor =
        Tensor(this)

    private fun dataIndex(indices: IntArray): Int {
        require(indices.size == rank) {
            "Expected $rank indices, got ${indices.size}"
        }

        var index = offset

        for (dimension in 0 until rank) {
            require(indices[dimension] in 0 until shape[dimension]) {
                "Index ${indices[dimension]} out of bounds " +
                        "for dimension $dimension with size ${shape[dimension]}"
            }

            index += indices[dimension] * strides[dimension]
        }

        return index
    }

    private fun dataIndex(flatIndex: Int): Int {
        var remainder = flatIndex
        var index = offset

        for (dimension in rank - 1 downTo 0) {
            val coordinate =
                remainder % shape[dimension]

            remainder /= shape[dimension]

            index += coordinate * strides[dimension]
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

            return MutableTensor(
                data = data,
                shape = shape,
                strides = contiguousStrides(shape),
                offset = 0,
            )
        }

        operator fun invoke(
            vararg shape: Int,
            init: (IntArray) -> Float,
        ): MutableTensor =
            invoke(shape.asList(), init)

        operator fun invoke(
            other: TensorLike,
        ): MutableTensor {
            val data = FloatArray(other.size) {
                other.getFlat(it)
            }

            return MutableTensor(
                data = data,
                shape = other.shape,
                strides = contiguousStrides(other.shape),
                offset = 0,
            )
        }

        fun scalar(value: Float): MutableTensor =
            MutableTensor(
                data = floatArrayOf(value),
                shape = emptyList(),
                strides = emptyList(),
                offset = 0,
            )

        private fun requireShape(shape: List<Int>) {
            require(shape.all { it > 0 }) {
                "Tensor dimensions must be positive"
            }
        }

        private fun sizeOf(shape: List<Int>): Int =
            shape.fold(1) { acc, dimension ->
                acc * dimension
            }

        private fun contiguousStrides(
            shape: List<Int>,
        ): List<Int> {
            var stride = 1
            val strides = MutableList(shape.size) { 0 }

            for (dimension in shape.size - 1 downTo 0) {
                strides[dimension] = stride
                stride *= shape[dimension]
            }

            return strides
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

    override fun slice(vararg slices: Slice): MutableTensor {
        require(slices.size <= rank) {
            "Expected at most $rank slices, got ${slices.size}"
        }

        var newOffset = offset

        val newShape = MutableList(rank) { 0 }
        val newStrides = MutableList(rank) { 0 }

        for (dimension in 0 until rank) {
            val slice =
                if (dimension < slices.size) {
                    slices[dimension]
                } else {
                    Slice()
                }

            val dimensionSize = shape[dimension]
            val end = slice.end ?: dimensionSize

            require(slice.start in 0..dimensionSize) {
                "Slice start ${slice.start} is out of bounds for dimension $dimension"
            }

            require(end in 0..dimensionSize) {
                "Slice end $end is out of bounds for dimension $dimension"
            }

            require(slice.start <= end) {
                "Slice start ${slice.start} must not be greater than end $end"
            }

            val resultSize =
                (end - slice.start + slice.step - 1) / slice.step

            require(resultSize > 0) {
                "Empty tensor slices are not currently supported"
            }

            newOffset += slice.start * strides[dimension]
            newShape[dimension] = resultSize
            newStrides[dimension] =
                strides[dimension] * slice.step
        }

        return MutableTensor(
            data = data,
            shape = newShape,
            strides = newStrides,
            offset = newOffset,
        )
    }
}