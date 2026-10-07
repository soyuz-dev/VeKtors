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

}