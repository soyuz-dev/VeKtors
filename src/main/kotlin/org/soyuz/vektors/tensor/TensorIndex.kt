package org.soyuz.vektors.tensor

sealed interface TensorIndex

data object All : TensorIndex

@JvmInline
value class Index(
    val value: Int,
) : TensorIndex

@JvmInline
value class Range(
    val value: IntRange,
) : TensorIndex
