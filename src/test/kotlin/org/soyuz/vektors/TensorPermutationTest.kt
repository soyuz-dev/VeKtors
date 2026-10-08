package org.soyuz.vektors

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class TensorPermutationTest {
    @Test
    fun `permutation changes axes correctly`() {
        val tensor = Tensor(2, 3, 4) { (x, y, z) ->
            (x * 100 + y * 10 + z).toFloat()
        }

        val transposed = tensor.T[2, 0, 1]

        assertEquals(listOf(4, 2, 3), transposed.shape)

        assertEquals(
            tensor.valueAt(1, 2, 3),
            transposed.valueAt(3, 1, 2),
        )
    }

    @Test
    fun `mutable permutation is a view`() {
        val tensor = MutableTensor(2, 3, 4) { (x, y, z) ->
            (x * 100 + y * 10 + z).toFloat()
        }

        val transposed = tensor.T[2, 0, 1]

        transposed.setValueAt(
            3,
            1,
            2,
            value = 69420f,
        )

        assertEquals(
            69420f,
            transposed.valueAt(3, 1, 2),
        )

        assertEquals(
            69420f,
            tensor.valueAt(1, 2, 3),
        )
    }

    @Test
    fun `slice after permutation still maps to correct backing element`() {
        val tensor = MutableTensor(2, 3, 4) { (x, y, z) ->
            (x * 100 + y * 10 + z).toFloat()
        }

        val view =
            tensor
                .T[2, 0, 1][
                Range(1..3),
                Index(1),
                Slice(0, 3, 2),
            ]

        assertEquals(listOf(3, 2), view.shape)

        assertEquals(101f, view.valueAt(0, 0))
        assertEquals(121f, view.valueAt(0, 1))

        assertEquals(102f, view.valueAt(1, 0))
        assertEquals(122f, view.valueAt(1, 1))

        assertEquals(103f, view.valueAt(2, 0))
        assertEquals(123f, view.valueAt(2, 1))

        view.setValueAt(
            2,
            1,
            value = 999f,
        )

        assertEquals(999f, tensor.valueAt(1, 2, 3))
    }
}
