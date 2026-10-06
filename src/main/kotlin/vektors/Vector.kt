package org.soyuz.vektors

import kotlin.math.acos
import kotlin.math.sqrt

class Vector private constructor(
    private val data: FloatArray,
) {
    val size: Int get() = data.size
    operator fun get(index: Int): Float = data[index]

    companion object {
        operator fun invoke(vararg values: Float): Vector {
            require(values.isNotEmpty()) { "Vector cannot be empty" }
            return Vector(values)
        }
    }

    operator fun plus(other: Vector): Vector {
        requireMatchSize(other)
        return Vector(FloatArray(size) { this[it] + other[it] })
    }

    operator fun minus(other: Vector): Vector {
        requireMatchSize(other)
        return Vector(FloatArray(size) { this[it] - other[it] })
    }

    operator fun times(other: Float) = Vector(FloatArray(size) { this[it] * other })
    operator fun div(other: Float) = Vector(FloatArray(size) { this[it] / other })

    operator fun Float.times(other: Vector) = Vector(FloatArray(size) { this * other[it] })

    fun sum() = data.sum()
    fun mean() = data.sum() / size
    fun map(transform: (Float) -> Float) = Vector(FloatArray(size) { transform(this[it]) })

    fun min() = data.min()
    fun max() = data.max()

    infix fun dot(other: Vector): Float {
        requireMatchSize(other)
        var sum = 0f
        for (i in 0 until size) {
            sum += this[i] * other[i]
        }
        return sum
    }

    fun distanceTo(other: Vector) = (this - other).magnitude()

    fun angleTo(other: Vector): Float {
        val thisMag = magnitude()
        val otherMag = other.magnitude()
        require(thisMag > 0 && otherMag > 0) { "Cannot compute angle to or from a zero vector" }
        val cosine = ((this dot other) / (thisMag * otherMag)).coerceIn(-1f, 1f)
        return acos(cosine)
    }

    fun projected(onto: Vector) =
        onto * (this dot onto) / onto.magnitudeSquared()

    fun magnitude() = sqrt(this dot this)

    fun magnitudeSquared() = this dot this

    fun normalized(): Vector = this / magnitude()

    fun lerp(to: Vector, t: Float) = this + (to - this) * t

    private fun requireMatchSize(other: Vector) =
        require(other.size == size) { "Vector dimensions must match: ${this.size} != ${other.size}" }
}