package org.soyuz.vektors

class Matrix private constructor(
    private val data: FloatArray,
    override val rows: Int,
    override val columns: Int,
) : MatrixLike {

    override operator fun get(row: Int, column: Int): Float =
        data[row * columns + column]

    companion object {
        operator fun invoke(vararg rows: FloatArray): Matrix {
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

            return Matrix(
                FloatArray(rows.size * columns) { index ->
                    rows[index / columns][index % columns]
                },
                rows.size,
                columns,
            )
        }

        operator fun invoke(other: MatrixLike): Matrix =
            Matrix(
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
}