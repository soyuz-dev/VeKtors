package org.soyuz.vektors

data class Slice(
    val start: Int = 0,
    val end: Int? = null,
    val step: Int = 1,
) {
    init {
        require(step > 0) {
            "Slice step must be positive"
        }
    }
}