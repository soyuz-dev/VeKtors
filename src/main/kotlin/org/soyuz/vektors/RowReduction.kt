package org.soyuz.vektors

class RowReduction internal constructor(
    val matrix: Matrix,
    pivotColumns: List<Int>,
) {
    val pivotColumns: List<Int> = pivotColumns.toList()

    val rank: Int
        get() = pivotColumns.size

    val freeColumns: List<Int>
        get() = (0 until matrix.columns).filter {
            it !in pivotColumns
        }
}
