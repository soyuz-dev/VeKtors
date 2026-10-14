package org.soyuz.vektors

import org.soyuz.vektors.matrix.Matrix
import org.soyuz.vektors.matrix.MatrixLike
import org.soyuz.vektors.tensor.Tensor
import org.soyuz.vektors.tensor.TensorLike
import org.soyuz.vektors.vector.Vector
import org.soyuz.vektors.vector.VectorLike

fun VectorLike.toTensor(): Tensor =
    Tensor(listOf(size)) { (i) ->
        this[i]
    }

fun MatrixLike.toTensor(): Tensor =
    Tensor(rows, columns) { (row, column) ->
        this[row, column]
    }

fun TensorLike.toVector(): Vector {
    require(rank == 1) {
        "Cannot convert tensor of rank $rank to Vector"
    }

    return Vector(
        *FloatArray(shape[0]) { i ->
            valueAt(i)
        },
    )
}

fun TensorLike.toMatrix(): Matrix {
    require(rank == 2) {
        "Cannot convert tensor of rank $rank to Matrix"
    }

    return Matrix(
        rows = shape[0],
        columns = shape[1],
    ) { row, column ->
        valueAt(row, column)
    }
}