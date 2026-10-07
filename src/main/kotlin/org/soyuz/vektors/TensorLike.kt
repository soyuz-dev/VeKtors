package org.soyuz.vektors

interface TensorLike {
    val shape: List<Int>

    val rank: Int
        get() = shape.size

    val size: Int
        get() = shape.fold(1) { acc, dimension ->
            acc * dimension
        }

    operator fun get(vararg indices: Int): Float

    fun getFlat(index: Int): Float

    operator fun plus(other: TensorLike): Tensor =
        elementwise(other) { a, b ->
            a + b
        }

    operator fun minus(other: TensorLike): Tensor =
        elementwise(other) { a, b ->
            a - b
        }

    operator fun times(other: TensorLike): Tensor =
        elementwise(other) { a, b ->
            a * b
        }

    operator fun div(other: TensorLike): Tensor =
        elementwise(other) { a, b ->
            a / b
        }

    operator fun times(other: Float): Tensor =
        Tensor.fromFlat(
            shape,
            FloatArray(size) {
                getFlat(it) * other
            }
        )

    operator fun div(other: Float): Tensor =
        Tensor.fromFlat(
            shape,
            FloatArray(size) {
                getFlat(it) / other
            }
        )

    fun sum(): Float {
        var sum = 0f

        for (i in 0 until size) {
            sum += getFlat(i)
        }

        return sum
    }

    fun mean(): Float =
        sum() / size

    fun min(): Float {
        var min = getFlat(0)

        for (i in 1 until size) {
            if (getFlat(i) < min) {
                min = getFlat(i)
            }
        }

        return min
    }

    fun max(): Float {
        var max = getFlat(0)

        for (i in 1 until size) {
            if (getFlat(i) > max) {
                max = getFlat(i)
            }
        }

        return max
    }

    fun map(transform: (Float) -> Float): Tensor =
        Tensor.fromFlat(
            shape,
            FloatArray(size) {
                transform(getFlat(it))
            }
        )

    fun reshape(vararg shape: Int): Tensor {
        require(shape.isNotEmpty()) {
            "Tensor must have at least one dimension"
        }

        require(shape.all { it > 0 }) {
            "Tensor dimensions must be positive"
        }

        val newSize =
            shape.fold(1) { acc, dimension ->
                acc * dimension
            }

        require(newSize == size) {
            "Cannot reshape tensor of size $size to shape ${shape.asList()}"
        }

        return Tensor.fromFlat(
            shape.asList(),
            FloatArray(size) {
                getFlat(it)
            }
        )
    }

    private fun broadcastShape(
        first: List<Int>,
        second: List<Int>,
    ): List<Int> {
        val rank = maxOf(first.size, second.size)

        return List(rank) { i ->
            val firstIndex = first.size - rank + i
            val secondIndex = second.size - rank + i

            val a =
                if (firstIndex >= 0) first[firstIndex]
                else 1

            val b =
                if (secondIndex >= 0) second[secondIndex]
                else 1

            require(a == b || a == 1 || b == 1) {
                "Cannot broadcast shapes $first and $second"
            }

            maxOf(a, b)
        }
    }

    private fun broadcastIndices(
        indices: IntArray,
        shape: List<Int>,
    ): IntArray {
        val offset = indices.size - shape.size

        return IntArray(shape.size) { dimension ->
            if (shape[dimension] == 1) {
                0
            } else {
                indices[dimension + offset]
            }
        }
    }

    private fun elementwise(
        other: TensorLike,
        operation: (Float, Float) -> Float,
    ): Tensor {
        val resultShape = broadcastShape(shape, other.shape)

        return Tensor(resultShape) { indices ->
            val leftIndices =
                broadcastIndices(indices, shape)

            val rightIndices =
                broadcastIndices(indices, other.shape)

            operation(
                get(*leftIndices),
                other.get(*rightIndices),
            )
        }
    }

    fun slice(vararg slices: Slice): Tensor {
        require(slices.size <= rank) {
            "Expected at most $rank slices, got ${slices.size}"
        }

        val resolved = List(rank) { dimension ->
            val slice =
                slices.getOrNull(dimension) ?: Slice()

            val end =
                slice.end ?: shape[dimension]

            require(slice.start in 0..shape[dimension]) {
                "Slice start ${slice.start} out of bounds for dimension $dimension"
            }

            require(end in 0..shape[dimension]) {
                "Slice end $end out of bounds for dimension $dimension"
            }

            require(slice.start <= end) {
                "Slice start must not exceed end"
            }

            ResolvedSlice(
                slice.start,
                end,
                slice.step,
            )
        }

        val resultShape = resolved.map {
            if (it.start == it.end) {
                0
            } else {
                (it.end - it.start + it.step - 1) / it.step
            }
        }

        require(resultShape.all { it > 0 }) {
            "Slices cannot produce an empty tensor"
        }

        return Tensor(resultShape) { resultIndices ->
            val sourceIndices =
                IntArray(rank) { dimension ->
                    val slice = resolved[dimension]

                    slice.start +
                            resultIndices[dimension] * slice.step
                }

            get(*sourceIndices)
        }
    }

    private data class ResolvedSlice(
        val start: Int,
        val end: Int,
        val step: Int,
    )

    fun select(
        dimension: Int,
        index: Int,
    ): Tensor {
        require(dimension in 0 until rank) {
            "Dimension out of bounds: $dimension"
        }

        require(index in 0 until shape[dimension]) {
            "Index $index out of bounds for dimension $dimension"
        }

        val resultShape =
            shape.filterIndexed { i, _ ->
                i != dimension
            }

        require(resultShape.isNotEmpty()) {
            "Cannot select the only dimension of a tensor"
        }

        return Tensor(resultShape) { resultIndices ->
            val sourceIndices = IntArray(rank)

            var resultDimension = 0

            for (sourceDimension in 0 until rank) {
                if (sourceDimension == dimension) {
                    sourceIndices[sourceDimension] = index
                } else {
                    sourceIndices[sourceDimension] =
                        resultIndices[resultDimension]

                    resultDimension++
                }
            }

            get(*sourceIndices)
        }
    }

}