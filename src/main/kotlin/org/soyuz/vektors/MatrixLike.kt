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


    fun rank(tolerance: Float = 1e-6f): Int =
        rref(tolerance).rank



    fun rref(tolerance: Float = 1e-6f): RowReduction {
        require(tolerance.isFinite() && tolerance >= 0f) {
            "Tolerance must be finite and non-negative"
        }

        val data = Array(rows) { row ->
            FloatArray(columns) { column ->
                this[row, column]
            }
        }

        require(data.all { row -> row.all { it.isFinite() } }) {
            "Matrix must contain only finite values"
        }

        var scale = 0f

        for (row in data) {
            for (value in row) {
                val magnitude = kotlin.math.abs(value)
                if (magnitude > scale) scale = magnitude
            }
        }

        if (scale == 0f) {
            return RowReduction(
                Matrix(rows, columns) { _, _ -> 0f },
                emptyList(),
            )
        }

        for (row in data) {
            for (column in row.indices) {
                row[column] /= scale
            }
        }

        val pivotColumns = mutableListOf<Int>()
        var pivotRow = 0

        for (column in 0 until columns) {
            if (pivotRow == rows) break

            var pivot = pivotRow

            for (row in pivotRow + 1 until rows) {
                if (
                    kotlin.math.abs(data[row][column]) >
                    kotlin.math.abs(data[pivot][column])
                ) {
                    pivot = row
                }
            }

            if (kotlin.math.abs(data[pivot][column]) <= tolerance) {
                continue
            }

            if (pivot != pivotRow) {
                val temporary = data[pivotRow]
                data[pivotRow] = data[pivot]
                data[pivot] = temporary
            }

            val pivotValue = data[pivotRow][column]

            for (k in column until columns) {
                data[pivotRow][k] /= pivotValue
            }

            data[pivotRow][column] = 1f

            for (row in 0 until rows) {
                if (row == pivotRow) continue

                val factor = data[row][column]
                data[row][column] = 0f

                for (k in column + 1 until columns) {
                    data[row][k] -= factor * data[pivotRow][k]
                }
            }

            pivotColumns.add(column)
            pivotRow++
        }

        return RowReduction(
            Matrix(rows, columns) { row, column ->
                val value = data[row][column]
                if (kotlin.math.abs(value) <= tolerance) 0f else value
            },
            pivotColumns,
        )
    }


}