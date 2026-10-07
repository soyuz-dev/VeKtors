package org.soyuz.vektors

class Tea<T : TensorLike> internal constructor(
    private val permute: (IntArray) -> T,
) {
    operator fun get(vararg axes: Int): T =
        permute(axes)
}