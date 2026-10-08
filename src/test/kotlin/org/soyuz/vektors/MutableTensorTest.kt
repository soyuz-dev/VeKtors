package org.soyuz.vektors

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class MutableTensorTest {
    @Test
    fun `mutable tensor can change values`() {
        val tensor = MutableTensor(2, 3) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        assertEquals(12f, tensor.valueAt(1, 2))

        tensor.setValueAt(
            1,
            2,
            value = 69f,
        )

        assertEquals(69f, tensor.valueAt(1, 2))
    }

    @Test
    fun `mutable indexed views share backing storage`() {
        val tensor = MutableTensor(2, 3) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val row = tensor[1]

        assertEquals(listOf(3), row.shape)
        assertEquals(12f, row.valueAt(2))

        row *= 2f

        assertEquals(24f, row.valueAt(2))
        assertEquals(24f, tensor.valueAt(1, 2))
    }

    @Test
    fun `mutable scalar views share backing storage`() {
        val tensor = MutableTensor(2, 3) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val scalar = tensor[1][2]

        assertEquals(emptyList<Int>(), scalar.shape)
        assertEquals(12f, scalar.valueAt())

        scalar *= 10f

        assertEquals(120f, scalar.valueAt())
        assertEquals(120f, tensor.valueAt(1, 2))
    }

    @Test
    fun `mutable slice view shares backing storage`() {
        val tensor = MutableTensor(6, 4) { (row, column) ->
            (row * 10 + column).toFloat()
        }

        val view = tensor[
            Slice(1, 6, 2),
            Slice(1, 4),
        ]

        assertEquals(listOf(3, 3), view.shape)

        view *= 10f

        assertEquals(110f, tensor.valueAt(1, 1))
        assertEquals(120f, tensor.valueAt(1, 2))
        assertEquals(130f, tensor.valueAt(1, 3))

        assertEquals(310f, tensor.valueAt(3, 1))
        assertEquals(320f, tensor.valueAt(3, 2))
        assertEquals(330f, tensor.valueAt(3, 3))

        assertEquals(510f, tensor.valueAt(5, 1))
        assertEquals(520f, tensor.valueAt(5, 2))
        assertEquals(530f, tensor.valueAt(5, 3))

        assertEquals(0f, tensor.valueAt(0, 0))
        assertEquals(20f, tensor.valueAt(2, 0))
        assertEquals(40f, tensor.valueAt(4, 0))
    }

    @Test
    fun `mixed mutable indexing produces writable view`() {
        val tensor = MutableTensor(4, 5, 6) { (x, y, z) ->
            (x * 100 + y * 10 + z).toFloat()
        }

        val view = tensor[
            Index(2),
            Range(1..3),
            Slice(0, 6, 2),
        ]

        assertEquals(listOf(3, 3), view.shape)

        view.setValueAt(
            1,
            2,
            value = 69420f,
        )

        assertEquals(69420f, view.valueAt(1, 2))
        assertEquals(69420f, tensor.valueAt(2, 2, 4))
    }
}
