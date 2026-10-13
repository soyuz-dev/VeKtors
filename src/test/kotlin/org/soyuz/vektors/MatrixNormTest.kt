
package org.soyuz.vektors

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test
import kotlin.math.sqrt

class MatrixNormTest {

    @Test
    fun `frobenius norm of three four matrix is five`() {
        val matrix = Matrix(
            floatArrayOf(3f, 0f),
            floatArrayOf(0f, 4f),
        )

        assertEquals(5f, matrix.frobeniusNorm(), 1e-6f)
    }

    @Test
    fun `one norm is maximum absolute column sum`() {
        val matrix = Matrix(
            floatArrayOf(1f, -2f, 3f),
            floatArrayOf(-4f, 5f, -6f),
        )

        assertEquals(9f, matrix.oneNorm(), 1e-6f)
    }

    @Test
    fun `infinity norm is maximum absolute row sum`() {
        val matrix = Matrix(
            floatArrayOf(1f, -2f, 3f),
            floatArrayOf(-4f, 5f, -6f),
        )

        assertEquals(15f, matrix.infinityNorm(), 1e-6f)
    }

    @Test
    fun `frobenius norm supports rectangular matrices`() {
        val matrix = Matrix(
            floatArrayOf(1f, 2f, 3f),
            floatArrayOf(4f, 5f, 6f),
        )

        assertEquals(
            sqrt(91.0).toFloat(),
            matrix.frobeniusNorm(),
            1e-5f,
        )
    }

    @Test
    fun `zero matrix has zero norms`() {
        val matrix = Matrix(3, 4) { _, _ -> 0f }

        assertEquals(0f, matrix.frobeniusNorm())
        assertEquals(0f, matrix.oneNorm())
        assertEquals(0f, matrix.infinityNorm())
    }

    @Test
    fun `identity matrix has expected norms`() {
        val matrix = Matrix.identity(4)

        assertEquals(2f, matrix.frobeniusNorm(), 1e-6f)
        assertEquals(1f, matrix.oneNorm())
        assertEquals(1f, matrix.infinityNorm())
    }

    @Test
    fun `norms are invariant under negation`() {
        val matrix = Matrix(
            floatArrayOf(1f, -2f),
            floatArrayOf(3f, -4f),
        )

        val negative = matrix * -1f

        assertEquals(
            matrix.frobeniusNorm(),
            negative.frobeniusNorm(),
        )
        assertEquals(
            matrix.oneNorm(),
            negative.oneNorm(),
        )
        assertEquals(
            matrix.infinityNorm(),
            negative.infinityNorm(),
        )
    }

    @Test
    fun `norms scale with scalar magnitude`() {
        val matrix = Matrix(
            floatArrayOf(1f, 2f),
            floatArrayOf(3f, 4f),
        )

        val scaled = matrix * -3f

        assertEquals(
            matrix.frobeniusNorm() * 3f,
            scaled.frobeniusNorm(),
            1e-5f,
        )
        assertEquals(
            matrix.oneNorm() * 3f,
            scaled.oneNorm(),
            1e-5f,
        )
        assertEquals(
            matrix.infinityNorm() * 3f,
            scaled.infinityNorm(),
            1e-5f,
        )
    }

    @Test
    fun `norms work with mutable matrices`() {
        val matrix = MutableMatrix(
            floatArrayOf(3f, 0f),
            floatArrayOf(0f, 4f),
        )

        assertEquals(5f, matrix.frobeniusNorm(), 1e-6f)

        matrix[0, 0] = 0f

        assertEquals(4f, matrix.frobeniusNorm(), 1e-6f)
        assertEquals(4f, matrix.oneNorm())
        assertEquals(4f, matrix.infinityNorm())
    }

    @Test
    fun `norms reject non finite entries`() {
        val matrix = Matrix(
            floatArrayOf(1f, Float.NaN),
            floatArrayOf(2f, 3f),
        )

        assertThrows(IllegalArgumentException::class.java) {
            matrix.frobeniusNorm()
        }
        assertThrows(IllegalArgumentException::class.java) {
            matrix.oneNorm()
        }
        assertThrows(IllegalArgumentException::class.java) {
            matrix.infinityNorm()
        }
    }

    @Test
    fun `frobenius norm handles large values`() {
        val matrix = Matrix(
            floatArrayOf(1e20f, 0f),
            floatArrayOf(0f, 1e20f),
        )

        assertEquals(
            sqrt(2.0).toFloat() * 1e20f,
            matrix.frobeniusNorm(),
            1e14f,
        )
    }
}
