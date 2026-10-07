package org.soyuz.vektors

class Vector private constructor(
    private val data: FloatArray,
) : VectorLike {
    override val size: Int
        get() = data.size

    override operator fun get(index: Int): Float =
        data[index]

    companion object {
        operator fun invoke(vararg values: Float): Vector {
            require(values.isNotEmpty()) {
                "Vector cannot be empty"
            }

            return Vector(values)
        }

        operator fun invoke(other: VectorLike): Vector =
            Vector(FloatArray(other.size) { other[it] })
    }
}