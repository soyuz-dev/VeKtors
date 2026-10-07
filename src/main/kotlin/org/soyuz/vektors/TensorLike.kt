package org.soyuz.vektors

interface TensorLike {
    val shape: List<Int>

    val rank: Int
        get() = shape.size

    val size: Int
        get() = shape.fold(1) { acc, dimension ->
            acc * dimension
        }

    operator fun get(index: Int): TensorLike

    fun valueAt(vararg indices: Int): Float {
        require(indices.size == rank) {
            "Expected $rank indices, got ${indices.size}"
        }

        return getFlat(flatIndex(indices))
    }

    fun getFlat(index: Int): Float

    operator fun plus(other: TensorLike): Tensor =
        elementwise(other) { a, b -> a + b }

    operator fun minus(other: TensorLike): Tensor =
        elementwise(other) { a, b -> a - b }

    operator fun times(other: TensorLike): Tensor =
        elementwise(other) { a, b -> a * b }

    operator fun div(other: TensorLike): Tensor =
        elementwise(other) { a, b -> a / b }

    operator fun times(scalar: Float): Tensor =
        Tensor.fromFlat(
            shape,
            FloatArray(size) { getFlat(it) * scalar },
        )

    operator fun div(scalar: Float): Tensor =
        Tensor.fromFlat(
            shape,
            FloatArray(size) { getFlat(it) / scalar },
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
        var minimum = getFlat(0)

        for (i in 1 until size) {
            if (getFlat(i) < minimum) {
                minimum = getFlat(i)
            }
        }

        return minimum
    }

    fun max(): Float {
        var maximum = getFlat(0)

        for (i in 1 until size) {
            if (getFlat(i) > maximum) {
                maximum = getFlat(i)
            }
        }

        return maximum
    }

    fun map(transform: (Float) -> Float): Tensor =
        Tensor.fromFlat(
            shape,
            FloatArray(size) { transform(getFlat(it)) },
        )

    fun reshape(vararg shape: Int): Tensor {
        require(shape.all { it > 0 }) {
            "Tensor dimensions must be positive"
        }

        val newShape = shape.toList()
        val newSize = newShape.fold(1) { acc, dimension ->
            acc * dimension
        }

        require(newSize == size) {
            "Cannot reshape tensor of size $size into shape $newShape"
        }

        return Tensor.fromFlat(
            newShape,
            FloatArray(size) { getFlat(it) },
        )
    }

    fun slice(vararg slices: Slice): TensorLike

    fun select(dimension: Int, index: Int): Tensor {
        require(rank > 0) {
            "Cannot select from a scalar tensor"
        }

        require(dimension in 0 until rank) {
            "Dimension $dimension is out of bounds for tensor of rank $rank"
        }

        require(index in 0 until shape[dimension]) {
            "Index $index is out of bounds for dimension $dimension with size ${shape[dimension]}"
        }

        val resultShape = shape.filterIndexed { i, _ ->
            i != dimension
        }

        return Tensor(resultShape) { indices ->
            val sourceIndices = IntArray(rank)
            var resultDimension = 0

            for (sourceDimension in 0 until rank) {
                sourceIndices[sourceDimension] =
                    if (sourceDimension == dimension) {
                        index
                    } else {
                        indices[resultDimension++]
                    }
            }

            valueAt(*sourceIndices)
        }
    }

    private fun elementwise(
        other: TensorLike,
        operation: (Float, Float) -> Float,
    ): Tensor {
        val resultShape = broadcastShape(shape, other.shape)

        return Tensor(resultShape) { indices ->
            operation(
                valueAt(*broadcastIndices(indices, shape)),
                other.valueAt(*broadcastIndices(indices, other.shape)),
            )
        }
    }

    private fun flatIndex(indices: IntArray): Int {
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

    private data class ResolvedSlice(
        val start: Int,
        val end: Int,
        val step: Int,
    )
}