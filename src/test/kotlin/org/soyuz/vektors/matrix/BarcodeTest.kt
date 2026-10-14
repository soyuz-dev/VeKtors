package org.soyuz.vektors.matrix

import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import org.soyuz.vektors.vector.Vector

class BarcodeTest {

    private fun assertReconstruction(
        matrix: MatrixLike,
        tolerance: Float = 1e-4f,
    ) {
        val qr = matrix.qr
        val reconstructed = qr.Q * qr.R

        Assertions.assertEquals(matrix.rows, reconstructed.rows)
        Assertions.assertEquals(matrix.columns, reconstructed.columns)

        for (row in 0 until matrix.rows) {
            for (column in 0 until matrix.columns) {
                Assertions.assertEquals(
                    matrix[row, column],
                    reconstructed[row, column],
                    tolerance,
                )
            }
        }
    }

    private fun assertOrthogonal(
        q: MatrixLike,
        tolerance: Float = 1e-4f,
    ) {
        val product = q.T * q

        for (row in 0 until product.rows) {
            for (column in 0 until product.columns) {
                Assertions.assertEquals(
                    if (row == column) 1f else 0f,
                    product[row, column],
                    tolerance,
                )
            }
        }
    }

    @Test
    fun `QR reconstructs square matrix`() {
        val matrix = Matrix(
            floatArrayOf(12f, -51f, 4f),
            floatArrayOf(6f, 167f, -68f),
            floatArrayOf(-4f, 24f, -41f),
        )

        assertReconstruction(matrix)
    }

    @Test
    fun `QR reconstructs tall matrix`() {
        val matrix = Matrix(
            floatArrayOf(1f, 2f),
            floatArrayOf(3f, 4f),
            floatArrayOf(5f, 6f),
            floatArrayOf(7f, 8f),
        )

        assertReconstruction(matrix)
    }

    @Test
    fun `Q has orthonormal columns`() {
        val matrix = Matrix(
            floatArrayOf(1f, 2f),
            floatArrayOf(3f, 4f),
            floatArrayOf(5f, 6f),
        )

        assertOrthogonal(matrix.qr.Q)
    }

    @Test
    fun `QR produces correct dimensions`() {
        val matrix = Matrix(5, 3) { row, column ->
            (row + column + 1).toFloat()
        }

        val qr = matrix.qr

        Assertions.assertEquals(5, qr.Q.rows)
        Assertions.assertEquals(3, qr.Q.columns)
        Assertions.assertEquals(3, qr.R.rows)
        Assertions.assertEquals(3, qr.R.columns)
    }

    @Test
    fun `R is upper triangular`() {
        val matrix = Matrix(
            floatArrayOf(1f, 2f, 3f),
            floatArrayOf(4f, 5f, 6f),
            floatArrayOf(7f, 8f, 10f),
            floatArrayOf(2f, 3f, 4f),
        )

        val r = matrix.qr.R

        for (row in 0 until r.rows) {
            for (column in 0 until row) {
                Assertions.assertEquals(0f, r[row, column])
            }
        }
    }

    @Test
    fun `QR reconstructs identity matrix`() {
        val matrix = Matrix.identity(4)

        assertReconstruction(matrix)
        assertOrthogonal(matrix.qr.Q)
    }

    @Test
    fun `QR handles zero matrix`() {
        val matrix = Matrix(4, 2) { _, _ -> 0f }

        assertReconstruction(matrix)
        assertOrthogonal(matrix.qr.Q)
    }

    @Test
    fun `QR handles rank deficient matrix`() {
        val matrix = Matrix(
            floatArrayOf(1f, 2f),
            floatArrayOf(2f, 4f),
            floatArrayOf(3f, 6f),
        )

        assertReconstruction(matrix)
        assertOrthogonal(matrix.qr.Q)
    }

    @Test
    fun `QR handles zero leading column`() {
        val matrix = Matrix(
            floatArrayOf(0f, 1f),
            floatArrayOf(0f, 2f),
            floatArrayOf(0f, 3f),
        )

        assertReconstruction(matrix)
        assertOrthogonal(matrix.qr.Q)
    }

    @Test
    fun `QR rejects wide matrices`() {
        val matrix = Matrix(2, 3) { _, _ -> 1f }

        Assertions.assertThrows(IllegalArgumentException::class.java) {
            matrix.qr
        }
    }

    @Test
    fun `QR rejects non finite matrices`() {
        val matrix = Matrix(
            floatArrayOf(1f, Float.NaN),
            floatArrayOf(2f, 3f),
        )

        Assertions.assertThrows(IllegalArgumentException::class.java) {
            matrix.qr
        }
    }

    @Test
    fun `QR does not mutate input`() {
        val matrix = MutableMatrix(
            floatArrayOf(1f, 2f),
            floatArrayOf(3f, 4f),
            floatArrayOf(5f, 6f),
        )

        val barcode = matrix.qr
        val reconstructed = barcode.Q * barcode.R

        matrix[0, 0] = 100f

        Assertions.assertEquals(1f, reconstructed[0, 0], 1e-4f)
        Assertions.assertEquals(100f, matrix[0, 0])
    }


    @Test
    fun `least squares solves consistent tall system`() {
        val a = Matrix(
            floatArrayOf(1f, 0f),
            floatArrayOf(0f, 1f),
            floatArrayOf(1f, 1f),
        )

        val b = Vector.Companion(2f, 3f, 5f)
        val x = a.solveLeastSquares(b)

        Assertions.assertEquals(2f, x[0], 1e-5f)
        Assertions.assertEquals(3f, x[1], 1e-5f)
    }

    @Test
    fun `least squares solves inconsistent system`() {
        val a = Matrix(
            floatArrayOf(1f, 0f),
            floatArrayOf(0f, 1f),
            floatArrayOf(1f, 1f),
        )

        val b = Vector.Companion(1f, 2f, 4f)
        val x = a.solveLeastSquares(b)

        Assertions.assertEquals(4f / 3f, x[0], 1e-5f)
        Assertions.assertEquals(7f / 3f, x[1], 1e-5f)
    }

    @Test
    fun `least squares residual is orthogonal to columns`() {
        val a = Matrix(
            floatArrayOf(1f, 0f),
            floatArrayOf(0f, 1f),
            floatArrayOf(1f, 1f),
        )

        val b = Vector.Companion(1f, 2f, 4f)
        val x = a.solveLeastSquares(b)

        val residual = a * x - b
        val normalResidual = a.T * residual

        for (i in 0 until normalResidual.size) {
            Assertions.assertEquals(0f, normalResidual[i], 1e-5f)
        }
    }

    @Test
    fun `least squares agrees with square system solving`() {
        val a = Matrix(
            floatArrayOf(2f, 1f),
            floatArrayOf(1f, 3f),
        )

        val b = Vector.Companion(5f, 7f)

        val direct = a.solve(b)
        val leastSquares = a.solveLeastSquares(b)

        for (i in 0 until b.size) {
            Assertions.assertEquals(direct[i], leastSquares[i], 1e-5f)
        }
    }

    @Test
    fun `barcode can solve multiple least squares systems`() {
        val a = Matrix(
            floatArrayOf(1f, 0f),
            floatArrayOf(0f, 1f),
            floatArrayOf(1f, 1f),
        )

        val barcode = a.qr

        val first = barcode.solveLeastSquares(
            Vector.Companion(2f, 3f, 5f),
        )

        val second = barcode.solveLeastSquares(
            Vector.Companion(4f, 1f, 5f),
        )

        Assertions.assertEquals(2f, first[0], 1e-5f)
        Assertions.assertEquals(3f, first[1], 1e-5f)

        Assertions.assertEquals(4f, second[0], 1e-5f)
        Assertions.assertEquals(1f, second[1], 1e-5f)
    }

    @Test
    fun `least squares rejects rank deficient system`() {
        val a = Matrix(
            floatArrayOf(1f, 0f),
            floatArrayOf(2f, 0f),
            floatArrayOf(3f, 0f),
        )

        Assertions.assertThrows(IllegalArgumentException::class.java) {
            a.solveLeastSquares(Vector.Companion(1f, 2f, 3f))
        }
    }

    @Test
    fun `least squares tolerance controls near singularity`() {
        val a = Matrix.diagonal(1f, 1e-5f)
        val b = Vector.Companion(2f, 3e-5f)

        val x = a.solveLeastSquares(b, tolerance = 1e-6f)

        Assertions.assertEquals(2f, x[0], 1e-4f)
        Assertions.assertEquals(3f, x[1], 1e-4f)

        Assertions.assertThrows(IllegalArgumentException::class.java) {
            a.solveLeastSquares(b, tolerance = 1e-4f)
        }
    }

    @Test
    fun `least squares rejects mismatched dimensions`() {
        val a = Matrix(3, 2) { row, column ->
            if (row == column) 1f else 0f
        }

        Assertions.assertThrows(IllegalArgumentException::class.java) {
            a.solveLeastSquares(Vector.Companion(1f, 2f))
        }
    }

    @Test
    fun `least squares rejects invalid tolerance`() {
        val a = Matrix.identity(2)
        val b = Vector.Companion(1f, 2f)

        Assertions.assertThrows(IllegalArgumentException::class.java) {
            a.solveLeastSquares(b, -1f)
        }

        Assertions.assertThrows(IllegalArgumentException::class.java) {
            a.solveLeastSquares(b, Float.NaN)
        }
    }

    @Test
    fun `least squares rejects non finite right hand side`() {
        val a = Matrix.identity(2)

        Assertions.assertThrows(IllegalArgumentException::class.java) {
            a.solveLeastSquares(Vector.Companion(1f, Float.NaN))
        }
    }

    @Test
    fun `least squares does not mutate original matrix`() {
        val a = MutableMatrix(
            floatArrayOf(1f, 0f),
            floatArrayOf(0f, 1f),
            floatArrayOf(1f, 1f),
        )

        val barcode = a.qr

        a[0, 0] = 100f

        val x = barcode.solveLeastSquares(Vector.Companion(2f, 3f, 5f))

        Assertions.assertEquals(2f, x[0], 1e-5f)
        Assertions.assertEquals(3f, x[1], 1e-5f)
        Assertions.assertEquals(100f, a[0, 0])
    }

}