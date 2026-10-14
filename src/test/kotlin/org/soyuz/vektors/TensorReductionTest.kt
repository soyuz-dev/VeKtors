
package org.soyuz.vektors

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

class TensorReductionTest {

    @Test
    fun `sum reduces first axis`() {
        val tensor = Tensor(2, 3) { (i, j) ->
            (i * 3 + j + 1).toFloat()
        }

        val result = tensor.sum(axis = 0)

        assertEquals(listOf(3), result.shape)
        assertEquals(5f, result.valueAt(0))
        assertEquals(7f, result.valueAt(1))
        assertEquals(9f, result.valueAt(2))
    }

    @Test
    fun `sum reduces second axis`() {
        val tensor = Tensor(2, 3) { (i, j) ->
            (i * 3 + j + 1).toFloat()
        }

        val result = tensor.sum(axis = 1)

        assertEquals(listOf(2), result.shape)
        assertEquals(6f, result.valueAt(0))
        assertEquals(15f, result.valueAt(1))
    }

    @Test
    fun `mean reduces selected axis`() {
        val tensor = Tensor(2, 3) { (i, j) ->
            (i * 3 + j + 1).toFloat()
        }

        val result = tensor.mean(axis = 1)

        assertEquals(listOf(2), result.shape)
        assertEquals(2f, result.valueAt(0))
        assertEquals(5f, result.valueAt(1))
    }

    @Test
    fun `min reduces selected axis`() {
        val tensor = Tensor(2, 3) { (i, j) ->
            (i * 3 + j + 1).toFloat()
        }

        val result = tensor.min(axis = 1)

        assertEquals(listOf(2), result.shape)
        assertEquals(1f, result.valueAt(0))
        assertEquals(4f, result.valueAt(1))
    }

    @Test
    fun `max reduces selected axis`() {
        val tensor = Tensor(2, 3) { (i, j) ->
            (i * 3 + j + 1).toFloat()
        }

        val result = tensor.max(axis = 1)

        assertEquals(listOf(2), result.shape)
        assertEquals(3f, result.valueAt(0))
        assertEquals(6f, result.valueAt(1))
    }

    @Test
    fun `reduction preserves remaining axis order`() {
        val tensor = Tensor(2, 3, 4) { (i, j, k) ->
            (i * 100 + j * 10 + k).toFloat()
        }

        val result = tensor.sum(axis = 1)

        assertEquals(listOf(2, 4), result.shape)

        for (i in 0 until 2) {
            for (k in 0 until 4) {
                var expected = 0f

                for (j in 0 until 3) {
                    expected += tensor.valueAt(i, j, k)
                }

                assertEquals(
                    expected,
                    result.valueAt(i, k),
                    1e-5f,
                )
            }
        }
    }

    @Test
    fun `reducing vector produces scalar tensor`() {
        val tensor = Tensor(3) { (i) ->
            (i + 1).toFloat()
        }

        val result = tensor.sum(axis = 0)

        assertEquals(emptyList<Int>(), result.shape)
        assertEquals(0, result.rank)
        assertEquals(6f, result.valueAt())
    }

    @Test
    fun `reducing scalar tensor throws`() {
        val scalar = Tensor.scalar(42f)

        assertThrows(IllegalArgumentException::class.java) {
            scalar.sum(axis = 0)
        }
    }

    @Test
    fun `reduction supports sliced views`() {
        val source = Tensor(6, 3) { (i, j) ->
            (i * 10 + j).toFloat()
        }

        val view = source[
            Slice(0, 6, 2),
            All,
        ]

        val result = view.sum(axis = 0)

        assertEquals(listOf(3), result.shape)
        assertEquals(60f, result.valueAt(0))
        assertEquals(63f, result.valueAt(1))
        assertEquals(66f, result.valueAt(2))
    }

    @Test
    fun `reduction supports permuted views`() {
        val source = Tensor(2, 3, 4) { (i, j, k) ->
            (i * 100 + j * 10 + k).toFloat()
        }

        val permuted = source.T[2, 0, 1]
        val result = permuted.sum(axis = 2)

        assertEquals(listOf(4, 2), result.shape)

        for (k in 0 until 4) {
            for (i in 0 until 2) {
                var expected = 0f

                for (j in 0 until 3) {
                    expected += source.valueAt(i, j, k)
                }

                assertEquals(
                    expected,
                    result.valueAt(k, i),
                    1e-5f,
                )
            }
        }
    }

    @Test
    fun `reduction supports mutable tensors`() {
        val tensor = MutableTensor(2, 3) { (i, j) ->
            (i * 3 + j + 1).toFloat()
        }

        val original = tensor.sum(axis = 1)

        tensor.setValueAt(0, 0, value = 100f)

        val updated = tensor.sum(axis = 1)

        assertEquals(6f, original.valueAt(0))
        assertEquals(105f, updated.valueAt(0))
        assertEquals(15f, updated.valueAt(1))
    }

    @Test
    fun `reduction leaves source unchanged`() {
        val tensor = MutableTensor(2, 2) { (i, j) ->
            (i * 2 + j + 1).toFloat()
        }

        tensor.mean(axis = 0)

        assertEquals(1f, tensor.valueAt(0, 0))
        assertEquals(2f, tensor.valueAt(0, 1))
        assertEquals(3f, tensor.valueAt(1, 0))
        assertEquals(4f, tensor.valueAt(1, 1))
    }

    @Test
    fun `invalid axes are rejected`() {
        val tensor = Tensor(2, 3) { 1f }

        assertThrows(IllegalArgumentException::class.java) {
            tensor.sum(axis = -1)
        }

        assertThrows(IllegalArgumentException::class.java) {
            tensor.mean(axis = 2)
        }

        assertThrows(IllegalArgumentException::class.java) {
            tensor.min(axis = 3)
        }

        assertThrows(IllegalArgumentException::class.java) {
            tensor.max(axis = -2)
        }
    }

    @Test
    fun `parameterless reductions remain unchanged`() {
        val tensor = Tensor(2, 3) { (i, j) ->
            (i * 3 + j + 1).toFloat()
        }

        assertEquals(21f, tensor.sum())
        assertEquals(3.5f, tensor.mean())
        assertEquals(1f, tensor.min())
        assertEquals(6f, tensor.max())
    }
}
