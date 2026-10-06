package org.soyuz.vektors

class Matrix private constructor(
    private val data: FloatArray,
    val rows: Int,
    val columns: Int,
) {
    init {
        require(rows > 0) { "Matrix must have at least one row" }
        require(columns > 0) { "Matrix must have at least one column" }
        require(data.size == rows * columns) {
            "Matrix data size must match dimensions: ${data.size} != ${rows * columns}"
        }
    }

    operator fun get(row: Int, column: Int): Float =
        data[row * columns + column]

    companion object {
        operator fun invoke(vararg rows: FloatArray): Matrix {
            require(rows.isNotEmpty()) { "Matrix cannot be empty" }

            val columns = rows.first().size
            require(columns > 0) { "Matrix cannot have empty rows" }
            require(rows.all { it.size == columns }) {
                "Matrix rows must have matching dimensions"
            }

            return Matrix(
                rows.flatMap { it.asList() }.toFloatArray(),
                rows.size,
                columns,
            )
        }
    }

    operator fun plus(other: Matrix): Matrix {
        requireMatchShape(other)
        return Matrix(
            FloatArray(data.size) { data[it] + other.data[it] },
            rows,
            columns,
        )
    }

    operator fun minus(other: Matrix): Matrix {
        requireMatchShape(other)
        return Matrix(
            FloatArray(data.size) { data[it] - other.data[it] },
            rows,
            columns,
        )
    }

    operator fun times(other: Float) =
        Matrix(
            FloatArray(data.size) { data[it] * other },
            rows,
            columns,
        )

    operator fun div(other: Float) =
        Matrix(
            FloatArray(data.size) { data[it] / other },
            rows,
            columns,
        )

    operator fun Float.times(other: Matrix) =
        Matrix(
            FloatArray(other.data.size) { this * other.data[it] },
            other.rows,
            other.columns,
        )

    operator fun times(other: Matrix): Matrix {
        require(columns == other.rows) {
            "Matrix dimensions must match for multiplication: ${rows}x${columns} != ${other.rows}x${other.columns}"
        }

        return Matrix(
            FloatArray(rows * other.columns) { index ->
                val row = index / other.columns
                val column = index % other.columns

                var sum = 0f
                for (k in 0 until columns) {
                    sum += this[row, k] * other[k, column]
                }
                sum
            },
            rows,
            other.columns,
        )
    }

    operator fun times(other: Vector): Vector {
        require(columns == other.size) {
            "Matrix columns must match vector size: $columns != ${other.size}"
        }

        return Vector(
            *FloatArray(rows) { row ->
                var sum = 0f
                for (column in 0 until columns) {
                    sum += this[row, column] * other[column]
                }
                sum
            },
        )
    }

    fun transpose(): Matrix =
        Matrix(
            FloatArray(data.size) { index ->
                val row = index / columns
                val column = index % columns
                this[column, row]
            },
            columns,
            rows,
        )

    private fun requireMatchShape(other: Matrix) =
        require(rows == other.rows && columns == other.columns) {
            "Matrix dimensions must match: ${rows}x${columns} != ${other.rows}x${other.columns}"
        }


    val T
        get() = transpose()
}
