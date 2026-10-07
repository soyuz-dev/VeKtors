package org.soyuz.vektors



class Tensor private constructor(
    private val data: FloatArray,
    shape: List<Int>,
    private val strides: List<Int>,
    private val offset: Int,
) : TensorLike {

    override val shape: List<Int> =
        shape.toList()

    override val T: Tea<Tensor>
        get() = Tea { axes ->
            permute(*axes)
        }

    override operator fun get(index: Int): Tensor {
        require(rank > 0) {
            "Cannot select from a scalar tensor"
        }

        require(index in 0 until shape[0]) {
            "Index $index is out of bounds for dimension 0 with size ${shape[0]}"
        }

        return Tensor(
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

    override operator fun get(
        vararg indices: TensorIndex,
    ): Tensor {
        require(indices.size <= rank) {
            "Expected at most $rank indices, got ${indices.size}"
        }

        var newOffset = offset
        val newShape = mutableListOf<Int>()
        val newStrides = mutableListOf<Int>()

        for (dimension in 0 until rank) {
            val index =
                indices.getOrElse(dimension) { All }

            val dimensionSize = shape[dimension]

            when (index) {
                All -> {
                    newShape += dimensionSize
                    newStrides += strides[dimension]
                }

                is Index -> {
                    require(index.value in 0 until dimensionSize) {
                        "Index ${index.value} is out of bounds for dimension $dimension with size $dimensionSize"
                    }

                    newOffset +=
                        index.value * strides[dimension]
                }

                is Range -> {
                    require(!index.value.isEmpty()) {
                        "Range must not be empty"
                    }

                    val start = index.value.first
                    val end = index.value.last

                    require(start >= 0 && end < dimensionSize) {
                        "Range ${index.value} is out of bounds for dimension $dimension with size $dimensionSize"
                    }

                    newOffset +=
                        start * strides[dimension]

                    newShape +=
                        end - start + 1

                    newStrides +=
                        strides[dimension]
                }

                is Slice -> {
                    val end =
                        index.end ?: dimensionSize

                    require(index.start in 0..dimensionSize) {
                        "Slice start ${index.start} is out of bounds for dimension $dimension"
                    }

                    require(end in 0..dimensionSize) {
                        "Slice end $end is out of bounds for dimension $dimension"
                    }

                    require(index.start <= end) {
                        "Slice start ${index.start} must not be greater than end $end"
                    }

                    val resultSize =
                        (end - index.start + index.step - 1) /
                                index.step

                    require(resultSize > 0) {
                        "Empty tensor slices are not currently supported"
                    }

                    newOffset +=
                        index.start * strides[dimension]

                    newShape +=
                        resultSize

                    newStrides +=
                        strides[dimension] * index.step
                }
            }
        }

        return Tensor(
            data = data,
            shape = newShape,
            strides = newStrides,
            offset = newOffset,
        )
    }

    override fun permute(vararg axes: Int): Tensor {
        requirePermutation(axes)

        return Tensor(
            data = data,
            shape = axes.map { shape[it] },
            strides = axes.map { strides[it] },
            offset = offset,
        )
    }

    private fun dataIndex(flatIndex: Int): Int {
        var remainder = flatIndex
        var index = offset

        for (dimension in rank - 1 downTo 0) {
            val coordinate =
                remainder % shape[dimension]

            remainder /= shape[dimension]

            index +=
                coordinate * strides[dimension]
        }

        return index
    }

    private fun requirePermutation(axes: IntArray) {
        require(axes.size == rank) {
            "Expected $rank axes, got ${axes.size}"
        }

        require(axes.all { it in 0 until rank }) {
            "Axes must be in 0 until $rank: ${axes.toList()}"
        }

        require(axes.toSet().size == rank) {
            "Axes must not contain duplicates: ${axes.toList()}"
        }
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

            return Tensor(
                data = data,
                shape = shape,
                strides = contiguousStrides(shape),
                offset = 0,
            )
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
                },
            )

        fun scalar(value: Float): Tensor =
            fromFlat(
                emptyList(),
                floatArrayOf(value),
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
                data = data.copyOf(),
                shape = shape,
                strides = contiguousStrides(shape),
                offset = 0,
            )
        }

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
            val strides =
                MutableList(shape.size) { 0 }

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
            val indices =
                IntArray(shape.size)

            for (dimension in shape.size - 1 downTo 0) {
                indices[dimension] =
                    remainder % shape[dimension]

                remainder /=
                    shape[dimension]
            }

            return indices
        }
    }
}