
package org.soyuz.vektors

import kotlin.math.sqrt

class Barcode private constructor(
    private val data: Array<DoubleArray>,
    private val tau: DoubleArray,
) {
    val rows: Int
        get() = data.size

    val columns: Int
        get() = tau.size

    val Q: Matrix
        get() {
            val q = Array(rows) { row ->
                DoubleArray(columns) { column ->
                    if (row == column) 1.0 else 0.0
                }
            }

            for (k in columns - 1 downTo 0) {
                val factor = tau[k]
                if (factor == 0.0) continue

                for (column in 0 until columns) {
                    var dot = q[k][column]

                    for (row in k + 1 until rows) {
                        dot += data[row][k] * q[row][column]
                    }

                    dot *= factor

                    q[k][column] -= dot

                    for (row in k + 1 until rows) {
                        q[row][column] -= data[row][k] * dot
                    }
                }
            }

            return Matrix(rows, columns) { row, column ->
                q[row][column].toFloat()
            }
        }

    val R: Matrix
        get() = Matrix(columns, columns) { row, column ->
            if (row <= column) {
                data[row][column].toFloat()
            } else {
                0f
            }
        }


    fun solveLeastSquares(
        b: VectorLike,
        tolerance: Float = 1e-6f,
    ): Vector {
        require(b.size == rows) {
            "Vector size ${b.size} must match matrix row count $rows"
        }
        require(tolerance.isFinite() && tolerance >= 0f) {
            "Tolerance must be finite and non-negative"
        }

        val m = rows
        val n = columns

        val y = DoubleArray(m) { b[it].toDouble() }

        require(y.all { it.isFinite() }) {
            "Right-hand side must contain only finite values"
        }

        // Apply Q^T to b using the stored Householder reflectors.
        for (k in 0 until n) {
            val factor = tau[k]
            if (factor == 0.0) continue

            var dot = y[k]

            for (row in k + 1 until m) {
                dot += data[row][k] * y[row]
            }

            dot *= factor
            y[k] -= dot

            for (row in k + 1 until m) {
                y[row] -= data[row][k] * dot
            }
        }

        // Use the largest diagonal magnitude as the reference scale.
        var scale = 0.0

        for (i in 0 until n) {
            val magnitude = kotlin.math.abs(data[i][i])
            if (magnitude > scale) scale = magnitude
        }

        require(scale > 0.0) {
            "Matrix does not have full column rank"
        }

        val threshold = tolerance.toDouble() * scale
        val x = DoubleArray(n)

        // Solve Rx = Q^T b by back-substitution.
        for (row in n - 1 downTo 0) {
            val diagonal = data[row][row]

            require(kotlin.math.abs(diagonal) > threshold) {
                "Matrix is rank-deficient or numerically singular"
            }

            var value = y[row]

            for (column in row + 1 until n) {
                value -= data[row][column] * x[column]
            }

            x[row] = value / diagonal
        }

        require(x.all { it.isFinite() && it.toFloat().isFinite() }) {
            "Least-squares solution cannot be represented with finite Float values"
        }

        return Vector(*FloatArray(n) { x[it].toFloat() })
    }


    companion object {
        operator fun invoke(matrix: MatrixLike): Barcode {
            require(matrix.rows >= matrix.columns) {
                "QR decomposition currently requires rows >= columns"
            }

            val m = matrix.rows
            val n = matrix.columns

            val data = Array(m) { row ->
                DoubleArray(n) { column ->
                    matrix[row, column].also {
                        require(it.isFinite()) {
                            "Matrix must contain only finite values"
                        }
                    }.toDouble()
                }
            }

            val tau = DoubleArray(n)

            for (k in 0 until n) {
                var normSquared = 0.0

                for (row in k until m) {
                    val value = data[row][k]
                    normSquared += value * value
                }

                val norm = sqrt(normSquared)

                if (norm == 0.0) {
                    tau[k] = 0.0
                    continue
                }

                val x0 = data[k][k]
                val alpha = if (x0 >= 0.0) -norm else norm
                val v0 = x0 - alpha

                var tailSquared = 0.0

                for (row in k + 1 until m) {
                    data[row][k] /= v0
                    tailSquared += data[row][k] * data[row][k]
                }

                val factor = 2.0 / (1.0 + tailSquared)
                tau[k] = factor

                for (column in k + 1 until n) {
                    var dot = data[k][column]

                    for (row in k + 1 until m) {
                        dot += data[row][k] * data[row][column]
                    }

                    dot *= factor
                    data[k][column] -= dot

                    for (row in k + 1 until m) {
                        data[row][column] -= data[row][k] * dot
                    }
                }

                data[k][k] = alpha
            }

            return Barcode(data, tau)
        }
    }
}
