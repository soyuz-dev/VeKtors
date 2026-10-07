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

    operator fun plus(other: TensorLike): Tensor {
        requireMatchShape(other)

        return Tensor.fromFlat(
            shape,
            FloatArray(size) {
                getFlat(it) + other.getFlat(it)
            }
        )
    }

    operator fun minus(other: TensorLike): Tensor {
        requireMatchShape(other)

        return Tensor.fromFlat(
            shape,
            FloatArray(size) {
                getFlat(it) - other.getFlat(it)
            }
        )
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

    private fun requireMatchShape(other: TensorLike) =
        require(shape == other.shape) {
            "Tensor shapes must match: $shape != ${other.shape}"
        }
}