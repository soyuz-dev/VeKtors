package org.soyuz.vektors

class MutableMatrix private constructor(
    private val data: FloatArray,
    override val rows: Int,
    override val columns: Int,
) : MatrixLike {

    override operator fun get(row: Int, column: Int): Float =
        data[row * columns + column]

    operator fun set(row: Int, column: Int, value: Float) {
        data[row * columns + column] = value
    }

    companion object {
        operator fun invoke(vararg rows: FloatArray): MutableMatrix {
            require(rows.isNotEmpty()) {
                "Matrix cannot be empty"
            }

            val columns = rows.first().size

            require(columns > 0) {
                "Matrix cannot have empty rows"
            }

            require(rows.all { it.size == columns }) {
                "Matrix rows must have matching dimensions"
            }

            return MutableMatrix(
                FloatArray(rows.size * columns) { index ->
                    rows[index / columns][index % columns]
                },
                rows.size,
                columns,
            )
        }

        operator fun invoke(other: MatrixLike): MutableMatrix =
            MutableMatrix(
                FloatArray(other.rows * other.columns) { index ->
                    other[
                        index / other.columns,
                        index % other.columns
                    ]
                },
                other.rows,
                other.columns,
            )
    }

    operator fun plusAssign(other: MatrixLike) {
        requireMatchShape(other)

        for (row in 0 until rows) {
            for (column in 0 until columns) {
                this[row, column] += other[row, column]
            }
        }
    }

    operator fun minusAssign(other: MatrixLike) {
        requireMatchShape(other)

        for (row in 0 until rows) {
            for (column in 0 until columns) {
                this[row, column] -= other[row, column]
            }
        }
    }

    operator fun timesAssign(other: Float) {
        for (i in data.indices) {
            data[i] *= other
        }
    }

    operator fun divAssign(other: Float) {
        for (i in data.indices) {
            data[i] /= other
        }
    }

    fun mapInPlace(transform: (Float) -> Float) {
        for (i in data.indices) {
            data[i] = transform(data[i])
        }
    }

    fun transposeInPlace() {
        require(rows == columns) {
            "In-place transpose requires a square matrix"
        }

        for (row in 0 until rows) {
            for (column in row + 1 until columns) {
                val temporary = this[row, column]

                this[row, column] = this[column, row]
                this[column, row] = temporary
            }
        }
    }

    fun toMatrix() =
        Matrix(this)

    private fun requireMatchShape(other: MatrixLike) =
        require(rows == other.rows && columns == other.columns) {
            "Matrix dimensions must match: " +
                    "${rows}x$columns != ${other.rows}x${other.columns}"
        }
}