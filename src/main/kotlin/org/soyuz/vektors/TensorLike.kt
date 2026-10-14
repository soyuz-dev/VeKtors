package org.soyuz.vektors

interface TensorLike {
    val shape: List<Int>

    val rank: Int
        get() = shape.size

    val size: Int
        get() = shape.fold(1) { acc, dimension ->
            acc * dimension
        }

    val T: Tea<out TensorLike>

    operator fun get(index: Int): TensorLike

    operator fun get(vararg indices: TensorIndex): TensorLike

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

        return if (resultShape.isEmpty()) {
            Tensor.scalar(
                valueAt(
                    *IntArray(rank) { sourceDimension ->
                        if (sourceDimension == dimension) index else 0
                    },
                ),
            )
        } else {
            Tensor(resultShape) { indices ->
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
    }

    fun permute(vararg axes: Int): TensorLike

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


    fun contract(
        other: TensorLike,
        axesA: IntArray,
        axesB: IntArray,
    ): Tensor {
        require(axesA.size == axesB.size) {
            "Both tensors must contract over the same number of axes"
        }

        require(axesA.all { it in 0 until rank }) {
            "Invalid contraction axes for tensor of rank $rank"
        }
        require(axesB.all { it in 0 until other.rank }) {
            "Invalid contraction axes for tensor of rank ${other.rank}"
        }

        require(axesA.toSet().size == axesA.size) {
            "Contraction axes must not contain duplicates"
        }
        require(axesB.toSet().size == axesB.size) {
            "Contraction axes must not contain duplicates"
        }

        for (i in axesA.indices) {
            require(shape[axesA[i]] == other.shape[axesB[i]]) {
                "Contracted dimensions must match: " +
                        "${shape[axesA[i]]} != ${other.shape[axesB[i]]}"
            }
        }

        val freeA = (0 until rank).filter { it !in axesA }
        val freeB = (0 until other.rank).filter { it !in axesB }

        val resultShape =
            freeA.map { shape[it] } +
                    freeB.map { other.shape[it] }

        val contractedShape = axesA.map { shape[it] }

        fun elementCount(dimensions: List<Int>): Int =
            dimensions.fold(1) { acc, dimension ->
                acc * dimension
            }

        fun decode(index: Int, dimensions: List<Int>): IntArray {
            var remainder = index
            val indices = IntArray(dimensions.size)

            for (axis in dimensions.indices.reversed()) {
                indices[axis] = remainder % dimensions[axis]
                remainder /= dimensions[axis]
            }

            return indices
        }

        val resultSize = elementCount(resultShape)
        val contractedSize = elementCount(contractedShape)

        val result = FloatArray(resultSize)

        val indicesA = IntArray(rank)
        val indicesB = IntArray(other.rank)

        for (flatIndex in 0 until resultSize) {
            val outputIndices = decode(flatIndex, resultShape)

            for (i in freeA.indices) {
                indicesA[freeA[i]] = outputIndices[i]
            }

            for (i in freeB.indices) {
                indicesB[freeB[i]] =
                    outputIndices[freeA.size + i]
            }

            var sum = 0.0

            for (contractedIndex in 0 until contractedSize) {
                val contractedIndices =
                    decode(contractedIndex, contractedShape)

                for (i in axesA.indices) {
                    indicesA[axesA[i]] = contractedIndices[i]
                    indicesB[axesB[i]] = contractedIndices[i]
                }

                sum +=
                    valueAt(*indicesA).toDouble() *
                            other.valueAt(*indicesB).toDouble()
            }

            result[flatIndex] = sum.toFloat()
        }

        return Tensor.fromFlat(resultShape, result)
    }

}