package org.soyuz.vektors

import kotlin.math.acos
import kotlin.math.sqrt

interface VectorLike {
    val size: Int

    operator fun get(index: Int): Float

    operator fun plus(other: VectorLike): Vector {
        requireMatchSize(other)
        return Vector(*FloatArray(size) { this[it] + other[it] })
    }

    operator fun minus(other: VectorLike): Vector {
        requireMatchSize(other)
        return Vector(*FloatArray(size) { this[it] - other[it] })
    }

    operator fun times(other: Float): Vector =
        Vector(*FloatArray(size) { this[it] * other })

    operator fun div(other: Float): Vector =
        Vector(*FloatArray(size) { this[it] / other })

    infix fun dot(other: VectorLike): Float {
        requireMatchSize(other)

        var sum = 0f
        for (i in 0 until size) {
            sum += this[i] * other[i]
        }
        return sum
    }

    fun sum(): Float {
        var sum = 0f
        for (i in 0 until size) {
            sum += this[i]
        }
        return sum
    }

    fun mean() = sum() / size

    fun min(): Float {
        var min = this[0]
        for (i in 1 until size) {
            if (this[i] < min) min = this[i]
        }
        return min
    }

    fun max(): Float {
        var max = this[0]
        for (i in 1 until size) {
            if (this[i] > max) max = this[i]
        }
        return max
    }

    fun map(transform: (Float) -> Float) =
        Vector(*FloatArray(size) { transform(this[it]) })

    fun magnitudeSquared() = this dot this

    fun magnitude() = sqrt(magnitudeSquared())

    fun normalized(): Vector =
        this / magnitude()

    fun distanceTo(other: VectorLike) =
        (this - other).magnitude()

    fun angleTo(other: VectorLike): Float {
        requireMatchSize(other)

        val thisMagnitude = magnitude()
        val otherMagnitude = other.magnitude()

        require(thisMagnitude > 0f && otherMagnitude > 0f) {
            "Cannot compute angle to or from a zero vector"
        }

        val cosine =
            ((this dot other) / (thisMagnitude * otherMagnitude))
                .coerceIn(-1f, 1f)

        return acos(cosine)
    }

    fun projected(onto: VectorLike): Vector {
        requireMatchSize(onto)

        val magnitudeSquared = onto.magnitudeSquared()
        require(magnitudeSquared > 0f) {
            "Cannot project onto a zero vector"
        }

        return onto * ((this dot onto) / magnitudeSquared)
    }

    fun lerp(to: VectorLike, t: Float): Vector {
        requireMatchSize(to)
        return this + (to - this) * t
    }

    private fun requireMatchSize(other: VectorLike) =
        require(other.size == size) {
            "Vector dimensions must match: $size != ${other.size}"
        }
}