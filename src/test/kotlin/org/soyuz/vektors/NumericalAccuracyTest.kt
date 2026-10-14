
package org.soyuz.vektors

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.soyuz.vektors.matrix.Matrix
import org.soyuz.vektors.matrix.MatrixLike
import org.soyuz.vektors.matrix.MutableMatrix
import org.soyuz.vektors.vector.Vector
import org.soyuz.vektors.vector.VectorLike
import kotlin.math.abs
import kotlin.math.sqrt

class NumericalAccuracyTest {

    private fun relativeError(
        expected: MatrixLike,
        actual: MatrixLike,
    ): Double {
        require(expected.rows == actual.rows)
        require(expected.columns == actual.columns)

        var errorSquared = 0.0
        var expectedSquared = 0.0

        for (row in 0 until expected.rows) {
            for (column in 0 until expected.columns) {
                val a = expected[row, column].toDouble()
                val b = actual[row, column].toDouble()
                val difference = a - b

                errorSquared += difference * difference
                expectedSquared += a * a
            }
        }

        val error = sqrt(errorSquared)
        val reference = sqrt(expectedSquared)

        return if (reference == 0.0) {
            error
        } else {
            error / reference
        }
    }

    private fun residualNorm(
        a: MatrixLike,
        x: VectorLike,
        b: VectorLike,
    ): Double {
        require(a.rows == b.size)
        require(a.columns == x.size)

        var squared = 0.0

        for (row in 0 until a.rows) {
            var value = -b[row].toDouble()

            for (column in 0 until a.columns) {
                value +=
                    a[row, column].toDouble() *
                            x[column].toDouble()
            }

            squared += value * value
        }

        return sqrt(squared)
    }

    private fun relativeNormalResidual(
        a: MatrixLike,
        x: VectorLike,
        b: VectorLike,
    ): Double {
        val residual = DoubleArray(a.rows) { row ->
            var value = -b[row].toDouble()

            for (column in 0 until a.columns) {
                value +=
                    a[row, column].toDouble() *
                            x[column].toDouble()
            }

            value
        }

        var normalSquared = 0.0

        for (column in 0 until a.columns) {
            var value = 0.0

            for (row in 0 until a.rows) {
                value +=
                    a[row, column].toDouble() * residual[row]
            }

            normalSquared += value * value
        }

        val matrixNorm = a.frobeniusNorm().toDouble()
        val residualNorm = sqrt(residual.sumOf { it * it })

        val denominator = matrixNorm * residualNorm

        return if (denominator == 0.0) {
            sqrt(normalSquared)
        } else {
            sqrt(normalSquared) / denominator
        }
    }

    @Test
    fun `LU reconstruction has small relative error`() {
        val a = Matrix(
            floatArrayOf(4f, 2f, 1f),
            floatArrayOf(3f, 5f, 2f),
            floatArrayOf(1f, 2f, 6f),
        )

        val pluto = a.lu

        val error = relativeError(
            pluto.P * a,
            pluto.L * pluto.U,
        )

        assertTrue(error < 1e-5, "LU error: $error")
    }

    @Test
    fun `LU reconstruction handles row pivoting`() {
        val a = Matrix(
            floatArrayOf(0f, 2f, 1f),
            floatArrayOf(3f, 1f, 4f),
            floatArrayOf(1f, 5f, 2f),
        )

        val pluto = a.lu

        val error = relativeError(
            pluto.P * a,
            pluto.L * pluto.U,
        )

        assertTrue(error < 1e-5, "LU pivoting error: $error")
    }

    @Test
    fun `LU reconstruction is accurate across scales`() {
        val original = Matrix(
            floatArrayOf(4f, 2f, 1f),
            floatArrayOf(3f, 5f, 2f),
            floatArrayOf(1f, 2f, 6f),
        )

        for (scale in floatArrayOf(1e-4f, 1f, 1e4f)) {
            val a = original * scale
            val pluto = a.lu

            val error = relativeError(
                pluto.P * a,
                pluto.L * pluto.U,
            )

            assertTrue(
                error < 1e-5,
                "LU error at scale $scale: $error",
            )
        }
    }

    @Test
    fun `QR reconstruction has small relative error`() {
        val a = Matrix(
            floatArrayOf(12f, -51f, 4f),
            floatArrayOf(6f, 167f, -68f),
            floatArrayOf(-4f, 24f, -41f),
        )

        val barcode = a.qr

        val error = relativeError(
            a,
            barcode.Q * barcode.R,
        )

        assertTrue(error < 1e-5, "QR error: $error")
    }

    @Test
    fun `QR reconstruction works for tall matrices`() {
        val a = Matrix(
            floatArrayOf(1f, 2f),
            floatArrayOf(3f, 4f),
            floatArrayOf(5f, 6f),
            floatArrayOf(7f, 8f),
        )

        val barcode = a.qr

        val error = relativeError(
            a,
            barcode.Q * barcode.R,
        )

        assertTrue(error < 1e-5, "Tall QR error: $error")
    }

    @Test
    fun `QR reconstruction is accurate across scales`() {
        val original = Matrix(
            floatArrayOf(2f, 1f),
            floatArrayOf(1f, 3f),
            floatArrayOf(4f, 2f),
            floatArrayOf(3f, 5f),
        )

        for (scale in floatArrayOf(1e-4f, 1f, 1e4f)) {
            val a = original * scale
            val barcode = a.qr

            val error = relativeError(
                a,
                barcode.Q * barcode.R,
            )

            assertTrue(
                error < 1e-5,
                "QR error at scale $scale: $error",
            )
        }
    }

    @Test
    fun `Q columns remain orthonormal`() {
        val a = Matrix(
            floatArrayOf(1f, 2f, 3f),
            floatArrayOf(4f, 5f, 6f),
            floatArrayOf(7f, 8f, 10f),
            floatArrayOf(2f, 3f, 5f),
        )

        val q = a.qr.Q

        val error = relativeError(
            Matrix.identity(q.columns),
            q.T * q,
        )

        assertTrue(
            error < 1e-5,
            "Q orthogonality error: $error",
        )
    }

    @Test
    fun `Q remains orthonormal for rank deficient matrix`() {
        val a = Matrix(
            floatArrayOf(1f, 2f),
            floatArrayOf(2f, 4f),
            floatArrayOf(3f, 6f),
        )

        val q = a.qr.Q

        val error = relativeError(
            Matrix.identity(q.columns),
            q.T * q,
        )

        assertTrue(
            error < 1e-5,
            "Rank deficient Q orthogonality error: $error",
        )
    }

    @Test
    fun `least squares residual is orthogonal to columns`() {
        val a = Matrix(
            floatArrayOf(1f, 0f),
            floatArrayOf(0f, 1f),
            floatArrayOf(1f, 1f),
        )

        val b = Vector(1f, 2f, 4f)
        val x = a.solveLeastSquares(b)

        val error = relativeNormalResidual(a, x, b)

        assertTrue(
            error < 1e-5,
            "Least squares orthogonality error: $error",
        )
    }

    @Test
    fun `least squares residual satisfies optimality`() {
        val a = Matrix(
            floatArrayOf(1f, 0f),
            floatArrayOf(0f, 1f),
            floatArrayOf(1f, 1f),
            floatArrayOf(2f, 1f),
        )

        val b = Vector(1f, 2f, 4f, 5f)
        val x = a.solveLeastSquares(b)

        val optimal = residualNorm(a, x, b)

        val alternatives = listOf(
            Vector(x[0] + 0.1f, x[1]),
            Vector(x[0] - 0.1f, x[1]),
            Vector(x[0], x[1] + 0.1f),
            Vector(x[0], x[1] - 0.1f),
        )

        for (candidate in alternatives) {
            val alternative = residualNorm(a, candidate, b)

            assertTrue(
                optimal <= alternative + 1e-6,
                "Candidate residual $alternative " +
                        "was smaller than $optimal",
            )
        }
    }

    @Test
    fun `least squares solution is accurate across scales`() {
        val a = Matrix(
            floatArrayOf(1f, 0f),
            floatArrayOf(0f, 1f),
            floatArrayOf(1f, 1f),
        )

        val b = Vector(1f, 2f, 4f)

        for (scale in floatArrayOf(1e-4f, 1f, 1e4f)) {
            val scaled = a * scale
            val scaledB = Vector(
                *FloatArray(b.size) { b[it] * scale },
            )

            val x = scaled.solveLeastSquares(scaledB)

            assertTrue(
                abs(x[0] - 4f / 3f) < 1e-4f,
                "Incorrect x[0] at scale $scale: ${x[0]}",
            )

            assertTrue(
                abs(x[1] - 7f / 3f) < 1e-4f,
                "Incorrect x[1] at scale $scale: ${x[1]}",
            )
        }
    }

    @Test
    fun `QR decomposition does not mutate mutable matrix`() {
        val a = MutableMatrix(
            floatArrayOf(2f, 1f),
            floatArrayOf(3f, 4f),
            floatArrayOf(5f, 6f),
        )

        val barcode = a.qr
        val original = org.soyuz.vektors.matrix.Matrix(a)

        a[0, 0] = 100f

        val error = relativeError(
            original,
            barcode.Q * barcode.R,
        )

        assertTrue(
            error < 1e-5,
            "QR snapshot error: $error",
        )
    }
}
