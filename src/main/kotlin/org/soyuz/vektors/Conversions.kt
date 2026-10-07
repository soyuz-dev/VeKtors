package org.soyuz.vektors

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