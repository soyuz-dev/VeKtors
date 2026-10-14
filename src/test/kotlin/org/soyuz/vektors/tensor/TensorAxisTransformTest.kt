package org.soyuz.vektors.tensor

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

class TensorAxisTransformTest {

    @Test
    fun `mapAxis transforms rows independently`() {
        val tensor = Tensor(2, 3) { (i, j) ->
            (i * 3 + j + 1).toFloat()
        }

        val result = tensor.mapAxis(1) { slice ->
            val mean = slice.mean()
            slice.map { it - mean }
        }

        assertEquals(listOf(2, 3), result.shape)

        for (i in 0 until 2) {
            assertEquals(-1f, result.valueAt(i, 0))
            assertEquals(0f, result.valueAt(i, 1))
            assertEquals(1f, result.valueAt(i, 2))
        }
    }

    @Test
    fun `mapAxis transforms columns independently`() {
        val tensor = Tensor(2, 3) { (i, j) ->
            (i * 3 + j + 1).toFloat()
        }

        val result = tensor.mapAxis(0) { slice ->
            slice.map { it * 2f }
        }

        for (i in 0 until 2) {
            for (j in 0 until 3) {
                assertEquals(
                    tensor.valueAt(i, j) * 2f,
                    result.valueAt(i, j),
                )
            }
        }
    }

    @Test
    fun `mapAxis preserves arbitrary tensor shape`() {
        val tensor = Tensor(2, 3, 4) { (i, j, k) ->
            (i * 100 + j * 10 + k).toFloat()
        }

        val result = tensor.mapAxis(1) { slice ->
            slice.map { it + 1f }
        }

        assertEquals(listOf(2, 3, 4), result.shape)
        assertEquals(124f, result.valueAt(1, 2, 3))
    }

    @Test
    fun `mapAxis callback runs once per slice`() {
        val tensor = Tensor(2, 3, 4) { 1f }
        var calls = 0

        tensor.mapAxis(1) { slice ->
            calls++
            slice
        }

        assertEquals(8, calls)
    }

    @Test
    fun `mapAxis rejects incorrectly shaped results`() {
        val tensor = Tensor(2, 3) { 1f }

        assertThrows(IllegalArgumentException::class.java) {
            tensor.mapAxis(1) {
                Tensor(2) { 0f }
            }
        }

        assertThrows(IllegalArgumentException::class.java) {
            tensor.mapAxis(1) {
                Tensor.scalar(1f)
            }
        }
    }

    @Test
    fun `reduceAxis computes range of each row`() {
        val tensor = Tensor(2, 3) { (i, j) ->
            (i * 10 + j * (i + 1)).toFloat()
        }

        val result = tensor.reduceAxis(1) {
            it.max() - it.min()
        }

        assertEquals(listOf(2), result.shape)
        assertEquals(2f, result.valueAt(0))
        assertEquals(4f, result.valueAt(1))
    }

    @Test
    fun `reduceAxis supports arbitrary custom operations`() {
        val tensor = Tensor(2, 3) { (i, j) ->
            (i * 3 + j + 1).toFloat()
        }

        val result = tensor.reduceAxis(1) { slice ->
            slice.getFlat(0) * slice.getFlat(2)
        }

        assertEquals(3f, result.valueAt(0))
        assertEquals(24f, result.valueAt(1))
    }

    @Test
    fun `reduceAxis invokes callback once per slice`() {
        val tensor = Tensor(2, 3, 4) { 1f }
        var calls = 0

        tensor.reduceAxis(1) {
            calls++
            it.sum()
        }

        assertEquals(8, calls)
    }

    @Test
    fun `axis transforms support rank one tensors`() {
        val tensor = Tensor(3) { (i) ->
            (i + 1).toFloat()
        }

        val mapped = tensor.mapAxis(0) {
            it.map { value -> value * 2f }
        }

        val reduced = tensor.reduceAxis(0) {
            it.sum()
        }

        assertEquals(listOf(3), mapped.shape)
        assertEquals(6f, mapped.valueAt(2))

        assertEquals(emptyList<Int>(), reduced.shape)
        assertEquals(6f, reduced.valueAt())
    }

    @Test
    fun `axis transforms support permuted views`() {
        val source = Tensor(2, 3, 4) { (i, j, k) ->
            (i * 100 + j * 10 + k).toFloat()
        }

        val permuted = source.T[2, 0, 1]

        val result = permuted.mapAxis(2) {
            it.map { value -> value + 5f }
        }

        assertEquals(listOf(4, 2, 3), result.shape)
        assertEquals(128f, result.valueAt(3, 1, 2))
    }

    @Test
    fun `axis transforms support sliced views`() {
        val source = Tensor(6, 3) { (i, j) ->
            (i * 10 + j).toFloat()
        }

        val view = source[Slice(0, 6, 2), All]

        val result = view.reduceAxis(0) { it.sum() }

        assertEquals(listOf(3), result.shape)
        assertEquals(60f, result.valueAt(0))
        assertEquals(63f, result.valueAt(1))
        assertEquals(66f, result.valueAt(2))
    }

    @Test
    fun `mapAxis returns independent immutable result`() {
        val source = MutableTensor(2, 2) { (i, j) ->
            (i * 2 + j + 1).toFloat()
        }

        val result = source.mapAxis(1) { it }

        source.setValueAt(0, 0, value = 100f)

        assertEquals(1f, result.valueAt(0, 0))
        assertEquals(100f, source.valueAt(0, 0))
    }
}
