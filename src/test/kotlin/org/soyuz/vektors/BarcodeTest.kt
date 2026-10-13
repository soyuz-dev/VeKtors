
package org.soyuz.vektors

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

class BarcodeTest {

    private fun assertReconstruction(
        matrix: MatrixLike,
        tolerance: Float = 1e-4f,
    ) {
        val qr = matrix.qr
        val reconstructed = qr.Q * qr.R

        assertEquals(matrix.rows, reconstructed.rows)
        assertEquals(matrix.columns, reconstructed.columns)

        for (row in 0 until matrix.rows) {
            for (column in 0 until matrix.columns) {
                assertEquals(
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
                assertEquals(
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

        assertEquals(5, qr.Q.rows)
        assertEquals(3, qr.Q.columns)
        assertEquals(3, qr.R.rows)
        assertEquals(3, qr.R.columns)
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
                assertEquals(0f, r[row, column])
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

        assertThrows(IllegalArgumentException::class.java) {
            matrix.qr
        }
    }

    @Test
    fun `QR rejects non finite matrices`() {
        val matrix = Matrix(
            floatArrayOf(1f, Float.NaN),
            floatArrayOf(2f, 3f),
        )

        assertThrows(IllegalArgumentException::class.java) {
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

        assertEquals(1f, reconstructed[0, 0], 1e-4f)
        assertEquals(100f, matrix[0, 0])
    }
}
