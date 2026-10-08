package org.soyuz.vektors

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test
import kotlin.math.sqrt

class TensorTest {
    @Test
    fun `basic tensor properties work`() {
        val tensor = Tensor(2, 3, 4) { (x, y, z) ->
            (x * 100 + y * 10 + z).toFloat()
        }

        assertEquals(listOf(2, 3, 4), tensor.shape)
        assertEquals(3, tensor.rank)
        assertEquals(24, tensor.size)

        assertEquals(0f, tensor.valueAt(0, 0, 0))
        assertEquals(123f, tensor.valueAt(1, 2, 3))
    }

    @Test
    fun `scalar tensor works`() {
        val scalar = Tensor.scalar(42f)

        assertEquals(emptyList<Int>(), scalar.shape)
        assertEquals(0, scalar.rank)
        assertEquals(1, scalar.size)
        assertEquals(42f, scalar.valueAt())
        assertEquals(42f, scalar.sum())
    }

    @Test
    fun `sum mean min and max work`() {
        val tensor = Tensor(2, 3) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        assertEquals(36f, tensor.sum())
        assertEquals(6f, tensor.mean())
        assertEquals(0f, tensor.min())
        assertEquals(12f, tensor.max())
    }

    @Test
    fun `map creates transformed tensor`() {
        val tensor = Tensor(2, 3) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val doubled = tensor.map { it * 2f }

        assertEquals(0f, doubled.valueAt(0, 0))
        assertEquals(4f, doubled.valueAt(0, 2))
        assertEquals(24f, doubled.valueAt(1, 2))
    }

    @Test
    fun `reshape preserves flattened values`() {
        val tensor = Tensor(2, 3) { (row, column) ->
            (row * 3 + column).toFloat()
        }

        val reshaped = tensor.reshape(3, 2)

        assertEquals(listOf(3, 2), reshaped.shape)

        assertEquals(0f, reshaped.valueAt(0, 0))
        assertEquals(1f, reshaped.valueAt(0, 1))
        assertEquals(2f, reshaped.valueAt(1, 0))
        assertEquals(3f, reshaped.valueAt(1, 1))
        assertEquals(4f, reshaped.valueAt(2, 0))
        assertEquals(5f, reshaped.valueAt(2, 1))
    }

    @Test
    fun `invalid reshape throws`() {
        val tensor = Tensor(2, 3) { 0f }

        assertThrows(IllegalArgumentException::class.java) {
            tensor.reshape(4, 2)
        }
    }

    @Test
    fun `elementwise addition works`() {
        val a = Tensor(2, 3) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val b = Tensor(2, 3) { (_, column) ->
            (column + 1).toFloat()
        }

        val result = a + b

        assertEquals(1f, result.valueAt(0, 0))
        assertEquals(3f, result.valueAt(0, 1))
        assertEquals(5f, result.valueAt(0, 2))

        assertEquals(11f, result.valueAt(1, 0))
        assertEquals(13f, result.valueAt(1, 1))
        assertEquals(15f, result.valueAt(1, 2))
    }

    @Test
    fun `broadcasting works`() {
        val a = Tensor(2, 3) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val b = Tensor(3) { (column) ->
            (column + 1).toFloat()
        }

        val result = a + b

        assertEquals(listOf(2, 3), result.shape)

        assertEquals(1f, result.valueAt(0, 0))
        assertEquals(3f, result.valueAt(0, 1))
        assertEquals(5f, result.valueAt(0, 2))

        assertEquals(11f, result.valueAt(1, 0))
        assertEquals(13f, result.valueAt(1, 1))
        assertEquals(15f, result.valueAt(1, 2))
    }

    @Test
    fun `scalar multiplication works`() {
        val tensor = Tensor(2, 2) { (row, column) ->
            (row * 2 + column).toFloat()
        }

        val result = tensor * 3f

        assertEquals(0f, result.valueAt(0, 0))
        assertEquals(3f, result.valueAt(0, 1))
        assertEquals(6f, result.valueAt(1, 0))
        assertEquals(9f, result.valueAt(1, 1))
    }

    @Test
    fun `three dimensional scalar field produces sensible reductions`() {
        val field = Tensor(DEPTH, HEIGHT, WIDTH) { (z, y, x) ->
            val nx =
                (x - WIDTH / 2f) /
                        (WIDTH / 2f)

            val ny =
                (y - HEIGHT / 2f) /
                        (HEIGHT / 2f)

            val nz =
                (z - DEPTH / 2f) /
                        (DEPTH / 2f)

            sqrt(
                nx * nx +
                        ny * ny +
                        nz * nz,
            )
        }

        assertEquals(
            listOf(DEPTH, HEIGHT, WIDTH),
            field.shape,
        )

        assertEquals(3, field.rank)
        assertEquals(
            DEPTH * HEIGHT * WIDTH,
            field.size,
        )

        assertEquals(
            0.071624234f,
            field.min(),
            FLOAT_TOLERANCE,
        )

        assertEquals(
            1.7320508f,
            field.max(),
            FLOAT_TOLERANCE,
        )

        assertEquals(
            0.9619557f,
            field.mean(),
            FLOAT_TOLERANCE,
        )
    }

    companion object {
        private const val DEPTH = 21
        private const val HEIGHT = 21
        private const val WIDTH = 41
        private const val FLOAT_TOLERANCE = 0.000001f
    }
}