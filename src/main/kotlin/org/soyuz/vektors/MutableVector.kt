package org.soyuz.vektors

class MutableVector private constructor(
    private val data: FloatArray,
) : VectorLike {
    override val size: Int
        get() = data.size

    override operator fun get(index: Int): Float =
        data[index]

    operator fun set(index: Int, value: Float) {
        data[index] = value
    }

    companion object {
        operator fun invoke(vararg values: Float): MutableVector {
            require(values.isNotEmpty()) {
                "Vector cannot be empty"
            }

            return MutableVector(values)
        }

        operator fun invoke(other: VectorLike): MutableVector =
            MutableVector(FloatArray(other.size) { other[it] })
    }

    operator fun plusAssign(other: VectorLike) {
        requireMatchSize(other)

        for (i in 0 until size) {
            data[i] += other[i]
        }
    }

    operator fun minusAssign(other: VectorLike) {
        requireMatchSize(other)

        for (i in 0 until size) {
            data[i] -= other[i]
        }
    }

    operator fun timesAssign(other: Float) {
        for (i in 0 until size) {
            data[i] *= other
        }
    }

    operator fun divAssign(other: Float) {
        for (i in 0 until size) {
            data[i] /= other
        }
    }

    fun mapInPlace(transform: (Float) -> Float) {
        for (i in 0 until size) {
            data[i] = transform(data[i])
        }
    }

    fun normalize() {
        val magnitude = magnitude()

        require(magnitude > 0f) {
            "Cannot normalize a zero vector"
        }

        for (i in 0 until size) {
            data[i] /= magnitude
        }
    }

    fun toVector() =
        Vector(this)

    private fun requireMatchSize(other: VectorLike) =
        require(other.size == size) {
            "Vector dimensions must match: $size != ${other.size}"
        }
}