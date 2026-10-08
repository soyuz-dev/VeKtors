package org.soyuz.vektors

import kotlin.math.abs

interface MatrixLike {
    val rows: Int
    val columns: Int

    operator fun get(row: Int, column: Int): Float

    operator fun plus(other: MatrixLike): Matrix {
        requireMatchShape(other)

        return Matrix(
            *Array(rows) { row ->
                FloatArray(columns) { column ->
                    this[row, column] + other[row, column]
                }
            }
        )
    }

    operator fun minus(other: MatrixLike): Matrix {
        requireMatchShape(other)

        return Matrix(
            *Array(rows) { row ->
                FloatArray(columns) { column ->
                    this[row, column] - other[row, column]
                }
            }
        )
    }

    operator fun times(other: Float): Matrix =
        Matrix(
            *Array(rows) { row ->
                FloatArray(columns) { column ->
                    this[row, column] * other
                }
            }
        )

    operator fun div(other: Float): Matrix =
        Matrix(
            *Array(rows) { row ->
                FloatArray(columns) { column ->
                    this[row, column] / other
                }
            }
        )

    operator fun times(other: MatrixLike): Matrix {
        require(columns == other.rows) {
            "Matrix dimensions must match for multiplication: " +
                    "${rows}x$columns != ${other.rows}x${other.columns}"
        }

        return Matrix(
            *Array(rows) { row ->
                FloatArray(other.columns) { column ->
                    var sum = 0f

                    for (k in 0 until columns) {
                        sum += this[row, k] * other[k, column]
                    }

                    sum
                }
            }
        )
    }

    operator fun times(other: VectorLike): Vector {
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
            }
        )
    }

    fun transpose(): Matrix =
        Matrix(
            *Array(columns) { row ->
                FloatArray(rows) { column ->
                    this[column, row]
                }
            }
        )

    val T: Matrix
        get() = transpose()

    fun map(transform: (Float) -> Float): Matrix =
        Matrix(
            *Array(rows) { row ->
                FloatArray(columns) { column ->
                    transform(this[row, column])
                }
            }
        )

    private fun requireMatchShape(other: MatrixLike) =
        require(rows == other.rows && columns == other.columns) {
            "Matrix dimensions must match: " +
                    "${rows}x$columns != ${other.rows}x${other.columns}"
        }

    fun trace(): Float {
        require(rows == columns) {
            "Trace is only defined for square matrices"
        }

        var trace = 0f

        for (i in 0 until rows) {
            trace += this[i, i]
        }

        return trace
    }


    val lu: Pluto
        get() = Pluto(this)

    fun determinant(): Float = lu.determinant

    fun inverse(): Matrix = lu.inverse

    fun solve(b: VectorLike): Vector = lu.solve(b)

    fun solve(b: MatrixLike): Matrix = lu.solve(b)

}