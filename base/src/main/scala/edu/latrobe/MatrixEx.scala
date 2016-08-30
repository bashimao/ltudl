/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2016 Matthias Langer (t3l@threelights.de)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package edu.latrobe

import breeze.linalg._
import breeze.storage.Zero
import scala.collection._
import scala.reflect._

object MatrixEx {

  @inline
  final def abs(src0: CSCMatrix[Real])
  : Unit = ArrayEx.abs(
    src0.data, 0, 1,
    src0.activeSize
  )

  @inline
  final def add(alpha: Real,
                dst0:  DenseMatrix[Real],
                src1:  DenseMatrix[Real])
  : Unit = {
    val data1 = src1.data
    val data0 = dst0.data

    require(dst0.rows == src1.rows)

    if (dst0.isTranspose == src1.isTranspose) {
      require(dst0.cols == src1.cols)
      ArrayEx.add(
        alpha,
        data0, dst0.offset, 1,
        data1, src1.offset, 1,
        dst0.size
      )
    }
    else {
      foreachColumn(dst0, src1)(
        (
          off0, stride0,
          off1, stride1
        ) => {
          ArrayEx.add(
            alpha,
            data0, off0, stride0,
            data1, off1, stride1,
            dst0.rows
          )
        }
      )
    }
  }

  @inline
  final def add(alpha: Real,
                dst0:  DenseMatrix[Real],
                beta:  Real,
                src1:  DenseMatrix[Real])
  : Unit = {
    val data1 = src1.data
    val data0 = dst0.data

    require(dst0.rows == src1.rows)

    if (dst0.isTranspose == src1.isTranspose) {
      require(dst0.cols == src1.cols)
      ArrayEx.add(
        alpha,
        data0, dst0.offset, 1,
        beta,
        data1, src1.offset, 1,
        dst0.size
      )
    }
    else {
      foreachColumn(dst0, src1)(
        (
          off0, stride0,
          off1, stride1
        ) => {
          ArrayEx.add(
            alpha,
            data0, off0, stride0,
            beta,
            data1, off1, stride1,
            dst0.rows
          )
        }
      )
    }
  }

  @inline
  final def add(dst0: DenseMatrix[Real],
                beta: Real,
                src1: DenseMatrix[Real])
  : Unit = {
    val data1 = src1.data
    val data0 = dst0.data

    require(dst0.rows == src1.rows)

    if (dst0.isTranspose == src1.isTranspose) {
      require(dst0.cols == src1.cols)
      ArrayEx.add(
        data0, dst0.offset, 1,
        beta,
        data1, src1.offset, 1,
        dst0.size
      )
    }
    else {
      foreachColumn(dst0, src1)(
        (
          off0, stride0,
          off1, stride1
        ) => {
          ArrayEx.add(
            data0, off0, stride0,
            beta,
            data1, off1, stride1,
            dst0.rows
          )
        }
      )
    }
  }

  @inline
  final def asVector[T](src0: DenseMatrix[T])
  : DenseVector[T] = src0.flatten(View.Require)

  @inline
  final def asOrToArray[T](src0: Matrix[T])
                          (implicit tagT: ClassTag[T])
  : Array[T] = src0 match {
    case src0: DenseMatrix[T] =>
      asOrToArray(src0)
    case src0: CSCMatrix[T] =>
      toArray(src0)
    case _ =>
      throw new MatchError(src0)
  }

  /**
    * Similar to toArray. But avoid allocation if possible.
    */
  @inline
  final def asOrToArray[T](src0: DenseMatrix[T])
                          (implicit tagT: ClassTag[T])
  : Array[T] = {
    if (src0.size == src0.data.length) {
      src0.data
    }
    else {
      toArray(src0)
    }
  }

  @inline
  final def asOrToDenseMatrix[T](src0: Matrix[T])
  : DenseMatrix[T] = src0 match {
    case src0: DenseMatrix[T] =>
      src0
    case src0: CSCMatrix[T] =>
      src0.toDense
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def asOrToDenseVector[T](src0: DenseMatrix[T])
  : DenseVector[T] = src0.flatten(View.Prefer)

  @inline
  final def bottomRow[T](src0: DenseMatrix[T])
  : Transpose[DenseVector[T]] = src0(-1, ::)

  final def classLabelsToSparseMatrix(noLabels:   Int,
                                      labelValue: Real,
                                      labels:     Seq[Int])
  : CSCMatrix[Real] = {
    val values = CSCMatrix.zeros[Real](noLabels, labels.length)
    values.reserve(values.cols)
    SeqEx.foreachPair(
      labels
    )((i, v) => values.update(v, i, labelValue))
    values
  }

  final def classLabelsToSparseMatrix(noLabels:   Int,
                                      labelValue: Real,
                                      labels:     Array[Int])
  : CSCMatrix[Real] = {
    val values = CSCMatrix.zeros[Real](noLabels, labels.length)
    values.reserve(values.cols)
    ArrayEx.foreachPair(
      labels
    )((i, v) => values.update(v, i, labelValue))
    values
  }

  @inline
  final def columns[T](src0: DenseMatrix[T])
  : Array[(Int, Int)] = {
    val result = new Array[(Int, Int)](src0.cols)
    foreachColumnPair(
      src0
    )((i, off0, stride0) => result(i) = (off0, stride0))
    result
  }

  @inline
  final def columns[T, U](src0: DenseMatrix[T],
                          src1: DenseMatrix[U])
  : Array[(Int, Int, Int, Int)] = {
    val result = new Array[(Int, Int, Int, Int)](src0.cols)
    foreachColumnPair(
      src0,
      src1
    )(
      (
        i,
        off0, stride0,
        off1, stride1
      ) => {
        result(i) = (
          off0, stride0,
          off1, stride1
        )
      }
    )
    result
  }

  @inline
  final def columns[T, U, V](src0: DenseMatrix[T],
                             src1: DenseMatrix[U],
                             src2: DenseMatrix[V])
  : Array[(Int, Int, Int, Int, Int, Int)] = {
    val result = new Array[(Int, Int, Int, Int, Int, Int)](src0.cols)
    foreachColumnPair(
      src0,
      src1,
      src2
    )(
      (
        i,
        off0, stride0,
        off1, stride1,
        off2, stride2
      ) => {
        result(i) = (
          off0, stride0,
          off1, stride1,
          off2, stride2
        )
      }
    )
    result
  }

  @inline
  final def columns[T, U, V, W](src0: DenseMatrix[T],
                                src1: DenseMatrix[U],
                                src2: DenseMatrix[V],
                                src3: DenseMatrix[W])
  : Array[(Int, Int, Int, Int, Int, Int, Int, Int)] = {
    val result = new Array[(Int, Int, Int, Int, Int, Int, Int, Int)](
      src0.cols
    )
    foreachColumnPair(
      src0,
      src1,
      src2,
      src3
    )(
      (
        i,
        off0, stride0,
        off1, stride1,
        off2, stride2,
        off3, stride3
      ) => {
        result(i) = (
          off0, stride0,
          off1, stride1,
          off2, stride2,
          off3, stride3
        )
      }
    )
    result
  }

  /*
  @inline
  final def columnsParallel[T](matrix0: DenseMatrix[T])
  : ParArray[(Int, Int)] = {
    ParArray.handoff(columns(matrix0))
  }

  @inline
  final def columnsParallel[T, U](matrix0: DenseMatrix[T],
                                  matrix1: DenseMatrix[U])
  : ParArray[(Int, Int, Int, Int)] = {
    ParArray.handoff(columns(matrix0, matrix1))
  }

  @inline
  final def columnsParallel[T, U, V](matrix0: DenseMatrix[T],
                                     matrix1: DenseMatrix[U],
                                     matrix2: DenseMatrix[V])
  : ParArray[(Int, Int, Int, Int, Int, Int)] = {
    ParArray.handoff(columns(matrix0, matrix1, matrix2))
  }

  @inline
  final def columnsParallel[T, U, V, W](matrix0: DenseMatrix[T],
                                        matrix1: DenseMatrix[U],
                                        matrix2: DenseMatrix[V],
                                        matrix3: DenseMatrix[W])
  : ParArray[(Int, Int, Int, Int, Int, Int, Int, Int)] = {
    ParArray.handoff(columns(matrix0, matrix1, matrix2, matrix3))
  }
  */

  /*
  @inline
  final def columnPairsParallel[T](matrix0: DenseMatrix[T])
  : ParArray[(Int, Int, Int)] = {
    ParArray.handoff(columnPairs(matrix0))
  }

  @inline
  final def columnPairsParallel[T, U](matrix0: DenseMatrix[T],
                                      matrix1: DenseMatrix[U])
  : ParArray[(Int, Int, Int, Int, Int)] = {
    ParArray.handoff(columnPairs(matrix0, matrix1))
  }

  @inline
  final def columnPairsParallel[T, U, V](matrix0: DenseMatrix[T],
                                         matrix1: DenseMatrix[U],
                                         matrix2: DenseMatrix[V])
  : ParArray[(Int, Int, Int, Int, Int, Int, Int)] = {
    ParArray.handoff(columnPairs(matrix0, matrix1, matrix2))
  }

  @inline
  final def columnPairsParallel[T, U, V, W](matrix0: DenseMatrix[T],
                                            matrix1: DenseMatrix[U],
                                            matrix2: DenseMatrix[V],
                                            matrix3: DenseMatrix[W])
  : ParArray[(Int, Int, Int, Int, Int, Int, Int, Int, Int)] = {
    ParArray.handoff(columnPairs(matrix0, matrix1, matrix2, matrix3))
  }
  */

  @inline
  final def columnVector[T](src0: DenseMatrix[T], index: Int)
  : DenseVector[T] = src0(::, index)

  @inline
  final def columnVectors[T](src0: DenseMatrix[T])
  : Array[DenseVector[T]] = {
    val result = new Array[DenseVector[T]](src0.cols)
    foreachColumnVectorPair(
      src0
    )((i, v0) => result(i) = v0)
    result
  }

  @inline
  final def columnVectors[T, U](src0: DenseMatrix[T],
                                src1: DenseMatrix[U])
  : Array[(DenseVector[T], DenseVector[U])] = {
    val result = new Array[(DenseVector[T], DenseVector[U])](src0.cols)
    foreachColumnVectorPair(
      src0,
      src1
    )((i, v0, v1) => result(i) = (v0, v1))
    result
  }

  @inline
  final def columnVectors[T, U, V](src0: DenseMatrix[T],
                                   src1: DenseMatrix[U],
                                   src2: DenseMatrix[V])
  : Array[(DenseVector[T], DenseVector[U], DenseVector[V])] = {
    val result = new Array[(DenseVector[T], DenseVector[U], DenseVector[V])](
      src0.cols
    )
    foreachColumnVectorPair(
      src0,
      src1,
      src2
    )((i, v0, v1, v2) => result(i) = (v0, v1, v2))
    result
  }

  @inline
  final def columnVectors[T, U, V, W](src0: DenseMatrix[T],
                                      src1: DenseMatrix[U],
                                      src2: DenseMatrix[V],
                                      src3: DenseMatrix[W])
  : Array[(DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W])] = {
    val result = new Array[(DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W])](
      src0.cols
    )
    foreachColumnVectorPair(
      src0,
      src1,
      src2,
      src3
    )((i, v0, v1, v2, v3) => result(i) = (v0, v1, v2, v3))
    result
  }

  /*
  @inline
  final def columnVectorsParallel[T](matrix0: DenseMatrix[T])
  : ParArray[DenseVector[T]] = {
    ParArray.handoff(columnVectors(matrix0))
  }

  @inline
  final def columnVectorsParallel[T, U](matrix0: DenseMatrix[T],
                                        matrix1: DenseMatrix[U])
  : ParArray[(DenseVector[T], DenseVector[U])] = {
    ParArray.handoff(columnVectors(matrix0, matrix1))
  }

  @inline
  final def columnVectorsParallel[T, U, V](matrix0: DenseMatrix[T],
                                           matrix1: DenseMatrix[U],
                                           matrix2: DenseMatrix[V])
  : ParArray[(DenseVector[T], DenseVector[U], DenseVector[V])] = {
    ParArray.handoff(columnVectors(matrix0, matrix1, matrix2))
  }

  @inline
  final def columnVectorsParallel[T, U, V, W](matrix0: DenseMatrix[T],
                                              matrix1: DenseMatrix[U],
                                              matrix2: DenseMatrix[V],
                                              matrix3: DenseMatrix[W])
  : ParArray[(DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W])] = {
    ParArray.handoff(columnVectors(matrix0, matrix1, matrix2, matrix3))
  }
  */

  @inline
  final def columnVectorPairs[T](src0: DenseMatrix[T])
  : Array[(Int, DenseVector[T])] = {
    val result = new Array[(Int, DenseVector[T])](src0.cols)
    foreachColumnVectorPair(
      src0
    )((i, v0) => result(i) = (i, v0))
    result
  }

  @inline
  final def columnVectorPairs[T, U](src0: DenseMatrix[T],
                                    src1: DenseMatrix[U])
  : Array[(Int, DenseVector[T], DenseVector[U])] = {
    val result = new Array[(Int, DenseVector[T], DenseVector[U])](src0.cols)
    foreachColumnVectorPair(
      src0,
      src1
    )((i, v0, v1) => result(i) = (i, v0, v1))
    result
  }

  @inline
  final def columnVectorPairs[T, U, V](src0: DenseMatrix[T],
                                       src1: DenseMatrix[U],
                                       src2: DenseMatrix[V])
  : Array[(Int, DenseVector[T], DenseVector[U], DenseVector[V])] = {
    val result = new Array[(Int, DenseVector[T], DenseVector[U], DenseVector[V])](
      src0.cols
    )
    foreachColumnVectorPair(
      src0,
      src1,
      src2
    )((i, v0, v1, v2) => result(i) = (i, v0, v1, v2))
    result
  }

  @inline
  final def columnVectorPairs[T, U, V, W](src0: DenseMatrix[T],
                                          src1: DenseMatrix[U],
                                          src2: DenseMatrix[V],
                                          src3: DenseMatrix[W])
  : Array[(Int, DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W])] = {
    val result = new Array[(Int, DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W])](
      src0.cols
    )
    foreachColumnVectorPair(
      src0,
      src1,
      src2,
      src3
    )((i, v0, v1, v2, v3) => result(i) = (i, v0, v1, v2, v3))
    result
  }

  /*
  @inline
  final def columnVectorPairsParallel[T](matrix0: DenseMatrix[T])
  : ParArray[(Int, DenseVector[T])] = {
    ParArray.handoff(columnVectorPairs(matrix0))
  }

  @inline
  final def columnVectorPairsParallel[T, U](matrix0: DenseMatrix[T],
                                            matrix1: DenseMatrix[U])
  : ParArray[(Int, DenseVector[T], DenseVector[U])] = {
    ParArray.handoff(columnVectorPairs(matrix0, matrix1))
  }

  @inline
  final def columnVectorPairsParallel[T, U, V](matrix0: DenseMatrix[T],
                                               matrix1: DenseMatrix[U],
                                               matrix2: DenseMatrix[V])
  : ParArray[(Int, DenseVector[T], DenseVector[U], DenseVector[V])] = {
    ParArray.handoff(columnVectorPairs(matrix0, matrix1, matrix2))
  }

  @inline
  final def columnVectorPairsParallel[T, U, V, W](matrix0: DenseMatrix[T],
                                                  matrix1: DenseMatrix[U],
                                                  matrix2: DenseMatrix[V],
                                                  matrix3: DenseMatrix[W])
  : ParArray[(Int, DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W])] = {
    ParArray.handoff(columnVectorPairs(matrix0, matrix1, matrix2, matrix3))
  }
  */

  @inline
  final def compact[T](dst0: CSCMatrix[T])
                      (implicit zero: Zero[T])
  : Unit = compactEx(
    dst0
  )(_ != zero.zero)

  @inline
  final def compactEx[T](dst0: CSCMatrix[T])
                        (zeroTest: T => Boolean)
  : Unit = {
    val indices0 = dst0.rowIndices
    val data0    = dst0.data
    val offsets0 = dst0.colPtrs
    var off0     = offsets0(0)
    var offR     = offsets0(0)
    var c        = 1
    while (c < offsets0.length) {
      val end0 = offsets0(c)
      while (off0 < end0) {
        if (zeroTest(data0(off0))) {
          indices0(offR) = indices0(off0)
          data0(offR)    = data0(off0)
          offR += 1
        }
        off0 += 1
      }
      offsets0(c) = offR
      c += 1
    }

    dst0.use(data0, offsets0, indices0, offR)
    dst0.compact()
  }

  @inline
  final def concatColumns[T](dst0: Array[T], offset0: Int, stride0: Int,
                             src1: DenseMatrix[T],
                             src2: Matrix[T])
  : Unit = src2 match {
    case src2: DenseMatrix[T] =>
      concatColumns(
        dst0, offset0, stride0,
        src1,
        src2
      )
    case src2: CSCMatrix[T] =>
      concatColumns(
        dst0, offset0, stride0,
        src1,
        src2
      )
    case _ =>
      throw new MatchError(src2)
  }

  @inline
  final def concatColumns[T](dst0: Array[T], offset0: Int, stride0: Int,
                             src1: DenseMatrix[T],
                             src2: DenseMatrix[T])
  : Unit = {
    require(!src1.isTranspose && !src2.isTranspose && src1.rows == src2.rows)
    copy(
      dst0, offset0, stride0,
      src1
    )
    copy(
      dst0, offset0 + stride0 * src1.size, stride0,
      src2
    )
  }

  @inline
  final def concatColumns[T](dst0: Array[T], offset0: Int, stride0: Int,
                             src1: DenseMatrix[T],
                             src2: CSCMatrix[T])
  : Unit = {
    require(!src1.isTranspose && src1.rows == src2.rows)
    copy(
      dst0, offset0, stride0,
      src1
    )
    copyActive(
      dst0, offset0 + stride0 * src1.size, stride0,
      src2
    )
  }

  @inline
  final def concatColumns[T](dst0: Array[T], offset0: Int, stride0: Int,
                             src1: CSCMatrix[T],
                             src2: DenseMatrix[T])
  : Unit = {
    require(!src2.isTranspose && src1.rows == src2.rows)
    copyActive(
      dst0, offset0, stride0,
      src1
    )
    copy(
      dst0, offset0 + stride0 * src1.size, stride0,
      src2
    )
  }

  /**
    * Vertical concatenation. (columns)
    */
  @inline
  final def concatColumns[T](src0: DenseMatrix[T],
                             src1: Matrix[T])
                            (implicit tagT: ClassTag[T])
  : Array[T] = src1 match {
    case src1: DenseMatrix[T] =>
      concatColumns(src0, src1)
    case src1: CSCMatrix[T] =>
      concatColumns(src0, src1)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def concatColumns[T](src0: DenseMatrix[T],
                             src1: DenseMatrix[T])
                            (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](src0.size + src1.size)
    copy(
      result, 0, 1,
      src0
    )
    copy(
      result, src0.size, 1,
      src1
    )
    result
  }

  @inline
  final def concatColumns[T](src0: DenseMatrix[T],
                             src1: CSCMatrix[T])
                            (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](src0.size + src1.size)
    copy(
      result, 0, 1,
      src0
    )
    copyActive(
      result, src0.size, 1,
      src1
    )
    result
  }

  @inline
  final def concatColumns[T](src0: CSCMatrix[T],
                             src1: DenseMatrix[T])
                            (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](src0.size + src1.size)
    copyActive(
      result, 0, 1,
      src0
    )
    copy(
      result, src0.size, 1,
      src1
    )
    result
  }

  // TODO: Slow!
  @inline
  final def concatColumns[T](src0: CSCMatrix[T],
                             src1: CSCMatrix[T])
                            (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : CSCMatrix[T] = {
    val result = CSCMatrix.zeros[T](src0.rows, src0.cols + src1.cols)
    result.reserve(src0.activeSize + src1.activeSize)
    foreachActivePair(src0)(
      (r, c, v1) => result.update(r, c, v1)
    )
    val cols0 = src0.cols
    foreachActivePair(src1)(
      (r, c, v1) => result.update(r, c + cols0, v1)
    )
    result
  }

  // TODO: Could be done faster.
  @inline
  final def concatColumnsDense[T](matrices: Array[DenseMatrix[T]])
                                 (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : DenseMatrix[T] = {
    val rows   = matrices.head.rows
    val cols   = ArrayEx.foldLeft(0, matrices)(_ + _.cols)
    val result = DenseMatrix.zeros[T](rows, cols)
    var c0     = 0
    ArrayEx.foreach(matrices)(m => {
      val c1 = c0 + m.cols
      result(::, c0 until c1) := m
      c0 = c1
    })
    result
  }

  // TODO: Could be done faster.
  @inline
  final def concatColumnsDense[T](matrices: Traversable[DenseMatrix[T]])
                                 (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : DenseMatrix[T] = {
    val rows   = matrices.head.rows
    val cols   = matrices.foldLeft(0)(_ + _.cols)
    val result = DenseMatrix.zeros[T](rows, cols)
    var c0     = 0
    matrices.foreach(m => {
      val c1 = c0 + m.cols
      result(::, c0 until c1) := m
      c0 = c1
    })
    result
  }

  // TODO: Slow!
  @inline
  final def concatColumnsSparse[T](matrices: Array[CSCMatrix[T]])
                                  (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : CSCMatrix[T] = {
    val rows   = matrices.head.rows
    val cols   = ArrayEx.foldLeft(0, matrices)(_ + _.cols)
    val result = CSCMatrix.zeros[T](rows, cols)
    var c0     = 0
    ArrayEx.foreach(matrices)(m => {
      val c1 = c0 + m.cols
      foreachActivePair(m)(
        (r, c, v) => result.update(r, c0 + c, v)
      )
      c0 = c1
    })
    result
  }

  // TODO: Slow!
  @inline
  final def concatColumnsSparse[T](matrices: Traversable[CSCMatrix[T]])
                                  (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : CSCMatrix[T] = {
    val rows   = matrices.head.rows
    val cols   = matrices.foldLeft(0)(_ + _.cols)
    val result = CSCMatrix.zeros[T](rows, cols)
    var c0     = 0
    matrices.foreach(m => {
      val c1 = c0 + m.cols
      foreachActivePair(m)(
        (r, c, v) => result.update(r, c0 + c, v)
      )
      c0 = c1
    })
    result
  }

  @inline
  final def concatRows[T](src0: DenseMatrix[T],
                          src1: Matrix[T])
                         (implicit tagT: ClassTag[T])
  : Array[T] = src1 match {
    case src1: DenseMatrix[T] =>
      concatRows(src0, src1)
    case src1: CSCMatrix[T] =>
      concatRows(src0, src1)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def concatRows[T](src0: DenseMatrix[T],
                          src1: DenseMatrix[T])
                         (implicit tagT: ClassTag[T])
  : Array[T] = {
    val data0 = src0.data
    val rows0 = src0.rows
    val cols0 = src0.cols
    val data1 = src1.data
    val rows1 = src1.rows
    val cols1 = src1.cols

    require(
      !src0.isTranspose &&
      !src1.isTranspose &&
      cols0 == cols1
    )

    val rowsR  = rows0 + rows1
    val result = new Array[T](rowsR * cols0)
    var i      = 0
    foreachColumn(src0, src1)(
      (
        off0, stride0,
        off1, stride1
      ) => {
        ArrayEx.set(
          result, i,    1,
          data0,  off0, stride0,
          rows0
        )
        i += rows0
        ArrayEx.set(
          result, i,    1,
          data1,  off1, stride1,
          rows1
        )
        i += rows1
      }
    )
    assume(i == result.length)
    result
  }

  /**
    * Vertical concatenation. (columns)
    */
  @inline
  final def concatRows[T](src0: DenseMatrix[T],
                          src1: CSCMatrix[T])
                         (implicit tagT: ClassTag[T])
  : Array[T] = {
    // Create shorthands for frequently used variables.
    val data0 = src0.data
    val rows0 = src0.rows
    val cols0 = src0.cols
    val rows1 = src1.rows

    require(
      !src0.isTranspose &&
      cols0 == src1.cols
    )

    // Allocate, fill result matrix and return.
    val rowsR  = rows0 + rows1
    val result = new Array[T](rowsR * cols0)
    var i      = 0
    foreachColumn(src0)(
      (off0, stride0) => {
        ArrayEx.set(
          result, i,    1,
          data0,  off0, stride0,
          rows0
        )
        i += rowsR
      }
    )
    foreachActivePair(src1)(
      (r, c, v1) => result(rows0 + rowsR * c + r) = v1
    )
    result
  }

  /**
    * Vertical concatenation. (columns)
    */
  @inline
  final def concatRows[T](src0: CSCMatrix[T],
                          src1: DenseMatrix[T])
                         (implicit tagT: ClassTag[T])
  : Array[T] = {
    // Create shorthands for frequently used variables.
    val rows0 = src0.rows
    val cols0 = src0.cols
    val rows1 = src1.rows
    val data1 = src1.data

    require(
      !src1.isTranspose &&
      cols0 == src1.cols
    )

    // Allocate, fill result matrix and return.
    val rowsR  = rows0 + rows1
    val result = new Array[T](rowsR * cols0)
    foreachActivePair(src1)(
      (r, c, v1) => result(rowsR * c + r) = v1
    )
    var i = rows0
    foreachColumn(src1)(
      (off1, stride1) => {
        ArrayEx.set(
          result, i,    1,
          data1,  off1, stride1,
          rows1
        )
        i += rowsR
      }
    )
    result
  }

  /**
    * Vertical concatenation. (columns)
    */
  @inline
  final def concatRows[T](src0: CSCMatrix[T],
                          src1: CSCMatrix[T])
                         (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : CSCMatrix[T] = {
    val used1    = src1.activeSize
    val offsets1 = src1.colPtrs
    val indices1 = src1.rowIndices
    val data1    = src1.data
    val rows1    = src1.rows
    val used0    = src0.activeSize
    val offsets0 = src0.colPtrs
    val indices0 = src0.rowIndices
    val data0    = src0.data
    val rows0    = src0.rows
    val cols0    = src0.cols

    require(
      rows0 == rows1 &&
      cols0 == src1.cols
    )

    // Allocate result buffers.
    val pointersR = new Array[Int](offsets0.length)
    val usedR     = used0 + used1
    val indicesR  = new Array[Int](usedR)
    val dataR     = new Array[T](usedR)
    var offR      = 0

    // Fill buffers.
    var off1 = offsets1(0)
    var off0 = offsets0(0)
    var c    = 1
    while (c < offsets0.length) {
      val end0  = offsets0(c)
      val used0 = end0 - off0
      val end1  = offsets1(c)
      val used1 = end1 - off1

      // matrix0
      ArrayEx.set(
        dataR, offR, 1,
        data0, off0, 1,
        used0
      )
      ArrayEx.set(
        indicesR, offR, 1,
        indices0, off0, 1,
        used0
      )
      off0 = end0
      offR += used0

      // matrix1
      ArrayEx.set(
        dataR, offR, 1,
        data1, off1, 1,
        used1
      )
      while (off1 < end1) {
        indicesR(offR) = rows0 + indices1(off1)
        offR += 1
        off1 += 1
      }

      // Set next col pointer.
      pointersR(c) = offR
      c += 1
    }
    assume(offR == usedR)

    new CSCMatrix(dataR, rows0 + rows1, cols0, pointersR, indicesR)
  }

  @inline
  final def contains[T](src0: DenseMatrix[T], value: T)
  : Boolean = exists(
    src0
  )(_ == value)

  @inline
  final def copy[T](dst0: Array[T],
                    src1: DenseMatrix[T])
  : Unit = {
    require(dst0.length == src1.size)
    copy(
      dst0, 0, 1,
      src1
    )
  }

  @inline
  final def copy[T](dst0: Array[T], offset0: Int, stride0: Int,
                    src1: DenseMatrix[T])
  : Unit = {
    val min1 = minor(src1)

    if (min1 == src1.majorStride) {
      ArrayEx.set(
        dst0,      offset0,     stride0,
        src1.data, src1.offset, 1,
        src1.size
      )
    }
    else {
      var off0 = offset0
      foreachMajor(src1)(off1 => {
        ArrayEx.set(
          dst0,      off0, stride0,
          src1.data, off1, 1,
          min1
        )
        off0 += stride0 * min1
      })
    }
  }

  @inline
  final def copyActive[T](dst0: Array[T], offset0: Int, stride0: Int,
                          src1:  CSCMatrix[T])
  : Unit = {
    val majorStride1 = src1.rows * stride0
    if (offset0 == 0) {
      foreachActivePair(
        src1
      )((r, c, v0) => dst0(c * majorStride1 + r) = v0)
    }
    else {
      foreachActivePair(
        src1
      )((r, c, v0) => dst0(c * majorStride1 + r + offset0) = v0)
    }
  }

  @inline
  final def exists[T](src0: DenseMatrix[T])
                     (fn: T => Boolean)
  : Boolean = {
    val data0   = src0.data
    var off0    = src0.offset
    val min0    = minor(src0)
    val maj0    = major(src0)
    val stride0 = src0.majorStride

    if (min0 == stride0) {
      ArrayEx.exists(
        data0, off0, 1,
        min0 * maj0
      )(fn)
    }
    else {
      val end0 = src0.offset + stride0 * maj0
      while (off0 != end0) {
        val result = ArrayEx.exists(
          src0.data, off0, 1,
          min0
        )(fn)
        if (result) {
          return true
        }
        off0 += stride0
      }
      false
    }
  }

  @inline
  final def divide(src0: Real,
                   dst1: CSCMatrix[Real])
  : Unit = ArrayEx.divide(
    src0,
    dst1.data, 0, 1,
    dst1.activeSize
  )

  /**
    * Syntax sugar for dot product.
    * (Actually, the result type of the generic implementation of Breeze is not
    * resolved properly into a Real by IntelliJ. That is because the type of
    * "That" is not determinable (why?!). So this even adds some value. YAY!)
    * Don't believe me? Replace the following line with:
    * (dv dot other) * Real.one
    * wait a second and the "*" will become red.
    *
    * The "Require" is not necessary, but I keep it in order to detect bottlenecks.
    */
  // TODO: Look at usages and think about implementing magnitude function.
  // TODO: Change to direct implementation in blas similar to axpy.
  @inline
  final def dot(src0: DenseMatrix[Real],
                src1: DenseMatrix[Real])
  : Real = foldLeft(
    Real.zero,
    src0,
    src1
  )(_ + _ * _)

  @inline
  final def extractColumnVector[T](src0: CSCMatrix[T], index: Int)
                                  (implicit zeroT: Zero[T])
  : SparseVector[T] = {
    val indices0 = src0.rowIndices
    val data0    = src0.data
    val off0     = src0.colPtrs(index)
    val end0     = src0.colPtrs(index + 1)
    val indicesR = ArrayEx.slice(indices0, off0, end0)
    val dataR    = ArrayEx.slice(data0,    off0, end0)
    new SparseVector(indicesR, dataR, src0.rows)
  }

  @inline
  final def extractColumnVectors[T](src0: CSCMatrix[T])
                                   (implicit zero: Zero[T])
  : Array[SparseVector[T]] = {
    val array = new Array[SparseVector[T]](src0.cols)
    var i = 0
    while (i < array.length) {
      array(i) = extractColumnVector(src0, i)
      i += 1
    }
    array
  }

  @inline
  final def extractColumnVectors[T](src0: CSCMatrix[T], slices: Range)
                                   (implicit zeroT: Zero[T])
  : Array[SparseVector[T]] = {
    val result = new Array[SparseVector[T]](slices.length)
    var i = 0
    slices.foreach(c => {
      result(i) = extractColumnVector(src0, c)
      i += 1
    })
    assume(i == result.length)
    result
  }

  @inline
  final def extractColumnVectorsEx[T](src0: CSCMatrix[T], slices: Range)
                                     (implicit classTag: ClassTag[T], zero: Zero[T])
  : CSCMatrix[T] = {
    val rows0    = src0.rows
    val offsets0 = src0.colPtrs
    val indices0 = src0.rowIndices
    val data0    = src0.data

    // Allocate memory.
    val colsR    = slices.length
    val offsetsR = new Array[Int](colsR + 1)
    val usedR    = slices.foldLeft(0)(
      (res, i) => res + offsets0(i + 1) - offsets0(i)
    )
    val indicesR = new Array[Int](usedR)
    val dataR    = new Array[T](usedR)

    // Copy items.
    var offR = 0
    var i    = 1
    slices.foreach(c => {
      val off0  = offsets0(c)
      val end0  = offsets0(c + 1)
      val used0 = end0 - off0

      ArrayEx.set(
        indicesR, offR, 1,
        indices0, off0, 1,
        used0
      )
      ArrayEx.set(
        dataR, offR, 1,
        data0, off0, 1,
        used0
      )
      offR += used0
      offsetsR(i) = offR
      i += 1
    })
    require(offR == usedR)

    new CSCMatrix(dataR, rows0, colsR, offsetsR, indicesR)
  }

  final def eye(rows: Int, cols: Int)
  : DenseMatrix[Real] = {
    val result = DenseMatrix.zeros[Real](rows, cols)
    val minDim = Math.min(rows, cols)
    // TODO: Could be done faster.
    var i = 0
    while (i < minDim) {
      result.update(i, i, Real.one)
      i += 1
    }
    result
  }

  @inline
  final def fill[T](rows: Int, cols: Int, value: T)
                   (implicit tagT: ClassTag[T])
  : DenseMatrix[T] = {
    val result = new DenseMatrix[T](rows, cols)
    ArrayEx.fill(
      result.data,
      value
    )
    result
  }

  @inline
  final def fill[T](rows: Int, cols: Int, distribution: Distribution[T])
                   (implicit tagT: ClassTag[T])
  : DenseMatrix[T] = {
    val result = new DenseMatrix[T](rows, cols)
    ArrayEx.fill(
      result.data,
      distribution
    )
    result
  }

  @inline
  final def fill[T](rows: Int, cols: Int)
                   (fn: => T)
                   (implicit tagT: ClassTag[T])
  : DenseMatrix[T] = {
    val result = new DenseMatrix[T](rows, cols)
    ArrayEx.fill(
      result.data
    )(fn)
    result
  }


  /**
    * Other method quite slow... Why are the Breeze people doing it so inefficient?
    * See also: numerics.sigmoid.inPlace(raw.values)
    * This is approximately 25% faster.
    */
  @inline
  final def fill[T](dst0: DenseMatrix[T])
                   (fn: => T)
  : Unit = {
    val data0   = dst0.data
    val min0    = minor(dst0)
    val stride0 = dst0.majorStride

    if (min0 == stride0) {
      ArrayEx.fill(
        data0, dst0.offset, 1,
        dst0.size
      )(fn)
    }
    else {
      foreachMajor(dst0)(
        ArrayEx.fill(
          data0, _, 1,
          min0
        )(fn)
      )
    }
  }

  @inline
  final def fill[T](dst0: DenseMatrix[T],
                    src1: Distribution[T])
  : Unit = {
    val data0   = dst0.data
    val min0    = minor(dst0)
    val stride0 = dst0.majorStride

    if (min0 == stride0) {
      ArrayEx.fill(
        data0, dst0.offset, 1,
        src1,
        dst0.size
      )
    }
    else {
      foreachMajor(dst0)(
        ArrayEx.fill(
          data0, _, 1,
          src1,
          min0
        )
      )
    }
  }

  @inline
  final def fill[T, U](dst0: DenseMatrix[T],
                       src1: DenseMatrix[U])
                      (fn: U => T)
  : Unit = {
    require(
      dst0.isTranspose == src1.isTranspose &&
      dst0.rows        == src1.rows        &&
      dst0.cols        == src1.cols
    )

    val data1   = src1.data
    val min1    = minor(src1)
    val stride1 = src1.majorStride

    val data0   = dst0.data
    val min0    = minor(dst0)
    val stride0 = dst0.majorStride

    if (
      min0 == stride0 &&
      min1 == stride1
    ) {
      ArrayEx.fill(
        data0, dst0.offset, 1,
        data1, src1.offset, 1,
        dst0.size
      )(fn)
    }
    else {
      foreachMajor(dst0, src1)(
        ArrayEx.fill(
          data0, _, 1,
          data1, _, 1,
          min0
        )(fn)
      )
    }
  }

  /**
    * Keeps only values where conditionFn evaluates true.
    */
  @inline
  final def filter[T](dst0: CSCMatrix[T])
                     (predicate: T => Boolean)
  : Unit = {
    val offsets0 = dst0.colPtrs
    val indices0 = dst0.rowIndices
    val data0    = dst0.data

    var off0 = offsets0(0)
    var offR = offsets0(0)
    var c    = 1
    while (c < offsets0.length) {
      val end0 = offsets0(c)

      while (off0 < end0) {
        val v0 = data0(off0)
        if (predicate(v0)) {
          indices0(offR) = indices0(off0)
          data0(offR)    = v0
          offR += 1
        }
        off0 += 1
      }

      offsets0(c) = offR
      c += 1
    }

    dst0.use(data0, offsets0, indices0, off0)
  }

  @inline
  final def foldLeft[T, U](src0: T,
                           src1: DenseMatrix[U])
                          (fn: (T, U) => T)
  : T = {
    var result = src0
    foreach(
      src1
    )(v1 => result = fn(result, v1))
    result
  }

  @inline
  final def foldLeft[T, U](src0: T,
                           src1: CSCMatrix[U])
                          (fn: (T, U) => T)
                          (implicit zeroU: Zero[U])
  : T = {
    var result = src0
    foreach(
      src1
    )(v1 => result = fn(result, v1))
    result
  }

  @inline
  final def foldLeft[T, U, V](src0:  T,
                              src1: DenseMatrix[U],
                              src2: Matrix[V])
                             (fn: (T, U, V) => T)
                             (implicit zero: Zero[V])
  : T = src2 match {
    case matrix2: DenseMatrix[V] =>
      foldLeft(src0, src1, matrix2)(fn)
    case matrix2: CSCMatrix[V] =>
      foldLeft(src0, src1, matrix2)(fn)
    case _ =>
      throw new MatchError(src2)
  }

  @inline
  final def foldLeft[T, U, V](src0: T,
                              src1: DenseMatrix[U],
                              src2: DenseMatrix[V])
                             (fn: (T, U, V) => T)
  : T = {
    var result = src0
    foreach(src1, src2)(
      (v1, v2) => result = fn(result, v1, v2)
    )
    result
  }

  @inline
  final def foldLeft[T, U, V](src0: T,
                              src1: DenseMatrix[U],
                              src2: CSCMatrix[V])
                             (fn: (T, U, V) => T)
                             (implicit zero: Zero[V])
  : T = {
    foldLeftEx(src0, src1, src2)(
      fn, fn(_, _, zero.zero)
    )
  }

  @inline
  final def foldLeftActive[T, U](src0:  T,
                                 src1: CSCMatrix[U])
                                (fn: (T, U) => T)
  : T = {
    var result = src0
    foreachActive(src1)(
      v1 => result = fn(result, v1)
    )
    result
  }

  @inline
  final def foldLeftActivePairs[T, U](src0:  T,
                                      src1: Matrix[U])
                                     (fn: (T, Int, Int, U) => T)
  : Unit = src1 match {
    case src1: DenseMatrix[U] =>
      foldLeftActivePairs(src0, src1)(fn)
    case src1: CSCMatrix[U] =>
      foldLeftActivePairs(src0, src1)(fn)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def foldLeftActivePairs[T, U](src0:  T,
                                      src1: DenseMatrix[U])
                                     (fn: (T, Int, Int, U) => T)
  : T = foldLeftPairs(src0, src1)(fn)

  @inline
  final def foldLeftActivePairs[T, U](src0:  T,
                                      src1: CSCMatrix[U])
                                     (fn: (T, Int, Int, U) => T)
  : T = {
    var result = src0
    foreachActivePair(src1)(
      (r, c, v1) => result = fn(result, r, c, v1)
    )
    result
  }

  @inline
  final def foldLeftColumns[T, U](src0:  T,
                                  src1: DenseMatrix[U])
                                 (fn: (T, Int, Int) => T)
  : T = {
    var result = src0
    foreachColumn(src1)(
      (off1, stride1) => result = fn(result, off1, stride1)
    )
    result
  }

  @inline
  final def foldLeftEx[T, U, V](src0: T,
                                src1: DenseMatrix[U],
                                src2: Matrix[V])
                               (fn0: (T, U, V) => T, fn1: (T, U) => T)
  : T = src2 match {
    case matrix2: DenseMatrix[V] =>
      foldLeft(src0, src1, matrix2)(fn0)
    case matrix2: CSCMatrix[V] =>
      foldLeftEx(src0, src1, matrix2)(fn0, fn1)
    case _ =>
      throw new MatchError(src2)
  }


  @inline
  final def foldLeftEx[T, U, V](src0: T,
                                src1: DenseMatrix[U],
                                src2: CSCMatrix[V])
                               (fn0: (T, U, V) => T, fn1: (T, U) => T)
  : T = {
    var result = src0
    foreachEx(src1, src2)(
      (v1, v2) => result = fn0(result, v1, v2),
      v1       => result = fn1(result, v1)
    )
    result
  }

  @inline
  final def foldLeftRows[T, U](src0: T,
                               src1: DenseMatrix[U])
                              (fn: (T, Int, Int) => T)
  : T = {
    var result = src0
    foreachRow(src1)(
      (off1, stride1) => result = fn(result, off1, stride1)
    )
    result
  }

  @inline
  final def foldLeftMinors[T, U](src0: T,
                                 src1: DenseMatrix[U])
                                (fn: (T, Int) => T)
  : T = {
    var result = src0
    foreachMajor(
      src1
    )(off1 => result = fn(result, off1))
    result
  }

  @inline
  final def foldLeftMinors[T, U, V](src0: T,
                                    src1: DenseMatrix[U],
                                    src2: DenseMatrix[V])
                                   (fn: (T, Int, Int) => T)
  : T = {
    var result = src0
    foreachMajor(
      src1,
      src2
    )((off1, off2) => result = fn(result, off1, off2))
    result
  }

  @inline
  final def foldLeftPairs[T, U](src0: T,
                                src1: DenseMatrix[U])
                               (fn: (T, Int, Int, U) => T)
  : T = {
    var result = src0
    foreachPair(
      src1
    )((r, c, v1) => result = fn(result, r, c, v1))
    result
  }

  @inline
  final def foreach[T](src0: Matrix[T])
                      (fn: T => Unit)
                      (implicit zero: Zero[T])
  : Unit = src0 match {
    case src0: DenseMatrix[T] =>
      foreach(
        src0
      )(fn)
    case src0: CSCMatrix[T] =>
      foreach(
        src0
      )(fn)
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def foreach[T](src0: DenseMatrix[T])
                      (fn: T => Unit)
  : Unit = {
    val data0   = src0.data
    val min0    = minor(src0)
    val stride0 = src0.majorStride

    if (stride0 == min0) {
      ArrayEx.foreach(
        data0, src0.offset, 1,
        src0.size
      )(fn)
    }
    else {
      foreachMajor(src0)(
        ArrayEx.foreach(
          data0, _, 1,
          min0
        )(fn)
      )
    }
  }

  @inline
  final def foreach[T](src0: CSCMatrix[T])
                      (fn: T => Unit)
                      (implicit zero: Zero[T])
  : Unit = {
    val offsets0 = src0.colPtrs
    val indices0 = src0.rowIndices
    val data0    = src0.data
    var off0     = offsets0(0)
    var c        = 1
    while (c < offsets0.length) {
      val end0 = offsets0(c)
      var r    = 0
      while (off0 < end0) {
        val index0 = indices0(off0)
        while (r < index0) {
          fn(zero.zero)
          r += 1
        }
        fn(data0(off0))
        r    += 1
        off0 += 1
      }
      c += 1
    }
  }

  @inline
  final def foreach[T, U](src0: DenseMatrix[T],
                          src1: Matrix[U])
                         (fn: (T, U) => Unit)
                         (implicit zero: Zero[U])
  : Unit = src1 match {
    case src1: DenseMatrix[U] =>
      foreach(
        src0,
        src1
      )(fn)
    case src1: CSCMatrix[U] =>
      foreach(
        src0,
        src1
      )(fn)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def foreach[T, U](src0: DenseMatrix[T],
                          src1: DenseMatrix[U])
                         (fn: (T, U) => Unit)
  : Unit = {
    val data1   = src1.data
    val min1    = minor(src1)
    val stride1 = src1.majorStride

    val data0   = src0.data
    val min0    = minor(src0)
    val stride0 = src0.majorStride

    require(
      src0.isTranspose == src1.isTranspose &&
      src0.rows        == src1.rows        &&
      src0.cols        == src1.cols
    )

    if (stride0 == min0 && stride1 == min1) {
      ArrayEx.foreach(
        data0, src0.offset, 1,
        data1, src1.offset, 1,
        src0.size
      )(fn)
    }
    else {
      foreachMajor(src0, src1)(
        ArrayEx.foreach(
          data0, _, 1,
          data1, _, 1,
          min0
        )(fn)
      )
    }
  }

  @inline
  final def foreach[T, U](src0: DenseMatrix[T],
                          src1: CSCMatrix[U])
                         (fn: (T, U) => Unit)
                         (implicit zero: Zero[U])
  : Unit = foreachEx(
    src0,
    src1
  )(fn, fn(_, zero.zero))

  @inline
  final def foreachActive[T](src0: Matrix[T])
                            (fn: T => Unit)
  : Unit = src0 match {
    case src0: DenseMatrix[T] =>
      foreachActive(
        src0
      )(fn)
    case src0: CSCMatrix[T] =>
      foreachActive(
        src0
      )(fn)
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def foreachActive[T](src0: DenseMatrix[T])
                            (fn: T => Unit)
  : Unit = foreach(
    src0
  )(fn)

  @inline
  final def foreachActive[T](src0: CSCMatrix[T])
                            (fn: T => Unit)
  : Unit = {
    val data0 = src0.data
    val used0 = src0.activeSize
    var i     = 0
    while (i < used0) {
      fn(data0(i))
      i += 1
    }
  }

  @inline
  final def foreachActivePair[T](src0: Matrix[T])
                                (fn: (Int, Int, T) => Unit)
  : Unit = src0 match {
    case src0: DenseMatrix[T] =>
      foreachActivePair(
        src0
      )(fn)
    case src0: CSCMatrix[T] =>
      foreachActivePair(
        src0
      )(fn)
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def foreachActivePair[T](src0: DenseMatrix[T])
                                (fn: (Int, Int, T) => Unit)
  : Unit = foreachPair(
    src0
  )(fn)

  @inline
  final def foreachActivePair[T](src0: CSCMatrix[T])
                                (fn: (Int, Int, T) => Unit)
  : Unit = {
    val cols0    = src0.cols
    val offsets0 = src0.colPtrs
    val indices0 = src0.rowIndices
    val data0    = src0.data
    var off0     = offsets0(0)
    var c0       = 0
    while (c0 < cols0) {
      val c1   = c0 + 1
      val end0 = offsets0(c1)
      while (off0 < end0) {
        fn(indices0(off0), c0, data0(off0))
        off0 += 1
      }
      c0 = c1
    }
  }

  @inline
  final def foreachColumn[T](src0: DenseMatrix[T])
                            (fn: (Int, Int) => Unit)
  : Unit = {
    val stride0 = src0.majorStride
    var off0    = src0.offset

    if (src0.isTranspose) {
      val end0 = src0.offset + src0.cols
      while (off0 < end0) {
        fn(off0, stride0)
        off0 += 1
      }
    }
    else {
      val end0 = src0.offset + src0.cols * stride0
      while (off0 < end0) {
        fn(off0, 1)
        off0 += stride0
      }
    }
  }

  @inline
  final def foreachColumn[T, U](src0: DenseMatrix[T],
                                src1: DenseMatrix[U])
                               (fn: (Int, Int, Int, Int) => Unit)
  : Unit = {
    require(
      src0.isTranspose == src1.isTranspose &&
      src0.cols        == src1.cols
    )

    val stride1 = src1.majorStride
    var off1    = src1.offset
    val stride0 = src0.majorStride
    var off0    = src0.offset

    if (src0.isTranspose) {
      val end0 = src0.offset + src0.cols
      while (off0 < end0) {
        fn(
          off0, stride0,
          off1, stride1
        )
        off1 += 1
        off0 += 1
      }
    }
    else {
      val end0 = src0.offset + src0.cols * stride0
      while (off0 < end0) {
        fn(
          off0, 1,
          off1, 1
        )
        off1 += stride1
        off0 += stride0
      }
    }
  }

  @inline
  final def foreachColumn[T, U, V](src0: DenseMatrix[T],
                                   src1: DenseMatrix[U],
                                   src2: DenseMatrix[V])
                                  (fn: (Int, Int, Int, Int, Int, Int) => Unit)
  : Unit = {
    require(
      src0.isTranspose == src1.isTranspose &&
      src0.cols        == src1.cols        &&
      src0.isTranspose == src2.isTranspose &&
      src0.cols        == src2.cols
    )

    val stride2 = src2.majorStride
    var off2    = src2.offset
    val stride1 = src1.majorStride
    var off1    = src1.offset
    val stride0 = src0.majorStride
    var off0    = src0.offset

    if (src0.isTranspose) {
      val end0 = src0.offset + src0.cols
      while (off0 < end0) {
        fn(
          off0, stride0,
          off1, stride1,
          off2, stride2
        )
        off2 += 1
        off1 += 1
        off0 += 1
      }
    }
    else {
      val end0 = src0.offset + src0.cols * stride0
      while (off0 < end0) {
        fn(
          off0, 1,
          off1, 1,
          off2, 1
        )
        off2 += stride2
        off1 += stride1
        off0 += stride0
      }
    }
  }

  @inline
  final def foreachColumn[T, U, V, W](src0: DenseMatrix[T],
                                      src1: DenseMatrix[U],
                                      src2: DenseMatrix[V],
                                      src3: DenseMatrix[W])
                                     (fn: (Int, Int, Int, Int, Int, Int, Int, Int) => Unit)
  : Unit = {
    require(
      src0.isTranspose == src1.isTranspose &&
      src0.cols        == src1.cols        &&
      src0.isTranspose == src2.isTranspose &&
      src0.cols        == src2.cols        &&
      src0.isTranspose == src3.isTranspose &&
      src0.cols        == src3.cols
    )

    val stride3 = src3.majorStride
    var off3    = src3.offset
    val stride2 = src2.majorStride
    var off2    = src2.offset
    val stride1 = src1.majorStride
    var off1    = src1.offset
    val stride0 = src0.majorStride
    var off0    = src0.offset

    if (src0.isTranspose) {
      val end0 = src0.offset + src0.cols
      while (off0 < end0) {
        fn(
          off0, stride0,
          off1, stride1,
          off2, stride2,
          off3, stride3
        )
        off3 += 1
        off2 += 1
        off1 += 1
        off0 += 1
      }
    }
    else {
      val end0 = src0.offset + src0.cols * stride0
      while (off0 < end0) {
        fn(
          off0, 1,
          off1, 1,
          off2, 1,
          off3, 1
        )
        off3 += stride3
        off2 += stride2
        off1 += stride1
        off0 += stride0
      }
    }
  }

  @inline
  final def foreachColumnPair[T](src0: DenseMatrix[T])
                                (fn: (Int, Int, Int) => Unit)
  : Unit = {
    val stride0 = src0.majorStride
    var off0    = src0.offset
    val n       = src0.cols
    var i       = 0

    if (src0.isTranspose) {
      while (i < n) {
        fn(
          i,
          off0, stride0
        )
        off0 += 1
        i    += 1
      }
    }
    else {
      while (i < n) {
        fn(
          i,
          off0, 1
        )
        off0 += stride0
        i    += 1
      }
    }
  }

  @inline
  final def foreachColumnPair[T, U](src0: DenseMatrix[T],
                                    src1: DenseMatrix[U])
                                   (fn: (Int, Int, Int, Int, Int) => Unit)
  : Unit = {
    require(
      src0.isTranspose == src1.isTranspose &&
      src0.cols        == src1.cols
    )

    val stride1 = src1.majorStride
    var off1    = src1.offset
    val stride0 = src0.majorStride
    var off0    = src0.offset
    val n       = src0.cols
    var i       = 0

    if (src0.isTranspose) {
      while (i < n) {
        fn(
          i,
          off0, stride0,
          off1, stride1
        )
        off1 += 1
        off0 += 1
        i    += 1
      }
    }
    else {
      while (i < n) {
        fn(
          i,
          off0, 1,
          off1, 1
        )
        off1 += stride1
        off0 += stride0
        i    += 1
      }
    }
  }

  @inline
  final def foreachColumnPair[T, U, V](src0: DenseMatrix[T],
                                       src1: DenseMatrix[U],
                                       src2: DenseMatrix[V])
                                      (fn: (Int, Int, Int, Int, Int, Int, Int) => Unit)
  : Unit = {
    require(
      src0.isTranspose == src1.isTranspose &&
      src0.cols        == src1.cols        &&
      src0.isTranspose == src2.isTranspose &&
      src0.cols        == src2.cols
    )

    val stride2 = src2.majorStride
    var off2    = src2.offset
    val stride1 = src1.majorStride
    var off1    = src1.offset
    val stride0 = src0.majorStride
    var off0    = src0.offset
    val n       = src0.cols
    var i       = 0

    if (src0.isTranspose) {
      while (i < n) {
        fn(
          i,
          off0, stride0,
          off1, stride1,
          off2, stride2
        )
        off2 += 1
        off1 += 1
        off0 += 1
        i    += 1
      }
    }
    else {
      while (i < n) {
        fn(
          i,
          off0, 1,
          off1, 1,
          off2, 1
        )
        off2 += stride2
        off1 += stride1
        off0 += stride0
        i    += 1
      }
    }
  }

  @inline
  final def foreachColumnPair[T, U, V, W](src0: DenseMatrix[T],
                                          src1: DenseMatrix[U],
                                          src2: DenseMatrix[V],
                                          src3: DenseMatrix[W])
                                         (fn: (Int, Int, Int, Int, Int, Int, Int, Int, Int) => Unit)
  : Unit = {
    require(
      src0.isTranspose == src1.isTranspose &&
      src0.cols        == src1.cols        &&
      src0.isTranspose == src2.isTranspose &&
      src0.cols        == src2.cols        &&
      src0.isTranspose == src3.isTranspose &&
      src0.cols        == src3.cols
    )

    val stride3 = src3.majorStride
    var off3    = src3.offset
    val stride2 = src2.majorStride
    var off2    = src2.offset
    val stride1 = src1.majorStride
    var off1    = src1.offset
    val stride0 = src0.majorStride
    var off0    = src0.offset
    val n       = src0.cols
    var i       = 0

    if (src0.isTranspose) {
      while (i < n) {
        fn(
          i,
          off0, stride0,
          off1, stride1,
          off2, stride2,
          off3, stride3
        )
        off3 += 1
        off2 += 1
        off1 += 1
        off0 += 1
        i    += 1
      }
    }
    else {
      while (i < n) {
        fn(
          i,
          off0, 1,
          off1, 1,
          off2, 1,
          off3, 1
        )
        off3 += stride3
        off2 += stride2
        off1 += stride1
        off0 += stride0
        i    += 1
      }
    }
  }

  @inline
  final def foreachColumnVector[T](src0: DenseMatrix[T])
                                  (fn: DenseVector[T] => Unit)
  : Unit = foreachColumn(src0)(
    (off0, stride0) => fn(
      new DenseVector(src0.data, off0, stride0, src0.rows)
    )
  )

  @inline
  final def foreachColumnVector[T, U](src0: DenseMatrix[T],
                                      src1: DenseMatrix[U])
                                     (fn: (DenseVector[T], DenseVector[U]) => Unit)
  : Unit = foreachColumn(src0, src1)(
    (
      off0, stride0,
      off1, stride1
    ) => fn(
      new DenseVector(src0.data, off0, stride0, src0.rows),
      new DenseVector(src1.data, off1, stride1, src1.rows)
    )
  )

  @inline
  final def foreachColumnVector[T, U, V](src0: DenseMatrix[T],
                                         src1: DenseMatrix[U],
                                         src2: DenseMatrix[V])
                                        (fn: (DenseVector[T], DenseVector[U], DenseVector[V]) => Unit)
  : Unit = foreachColumn(src0, src1, src2)(
    (
      off0, stride0,
      off1, stride1,
      off2, stride2
    ) => fn(
      new DenseVector(src0.data, off0, stride0, src0.rows),
      new DenseVector(src1.data, off1, stride1, src1.rows),
      new DenseVector(src2.data, off2, stride2, src2.rows)
    )
  )

  @inline
  final def foreachColumnVector[T, U, V, W](src0: DenseMatrix[T],
                                            src1: DenseMatrix[U],
                                            src2: DenseMatrix[V],
                                            src3: DenseMatrix[W])
                                           (fn: (DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W]) => Unit)
  : Unit = foreachColumn(src0, src1, src2, src3)(
    (
      off0, stride0,
      off1, stride1,
      off2, stride2,
      off3, stride3
    ) => fn(
      new DenseVector(src0.data, off0, stride0, src0.rows),
      new DenseVector(src1.data, off1, stride1, src1.rows),
      new DenseVector(src2.data, off2, stride2, src2.rows),
      new DenseVector(src3.data, off3, stride3, src3.rows)
    )
  )

  @inline
  final def foreachColumnVectorPair[T](src0: DenseMatrix[T])
                                      (fn: (Int, DenseVector[T]) => Unit)
  : Unit = foreachColumnPair(src0)(
    (
      i,
      off0, stride0
    ) => fn(
      i,
      new DenseVector(src0.data, off0, stride0, src0.rows)
    )
  )

  @inline
  final def foreachColumnVectorPair[T, U](src0: DenseMatrix[T],
                                          src1: DenseMatrix[U])
                                         (fn: (Int, DenseVector[T], DenseVector[U]) => Unit)
  : Unit = foreachColumnPair(src0, src1)(
    (
      i,
      off0, stride0,
      off1, stride1
    ) => fn(
      i,
      new DenseVector(src0.data, off0, stride0, src0.rows),
      new DenseVector(src1.data, off1, stride1, src1.rows)
    )
  )

  @inline
  final def foreachColumnVectorPair[T, U, V](src0: DenseMatrix[T],
                                             src1: DenseMatrix[U],
                                             src2: DenseMatrix[V])
                                            (fn: (Int, DenseVector[T], DenseVector[U], DenseVector[V]) => Unit)
  : Unit = foreachColumnPair(src0, src1, src2)(
    (
      i,
      off0, stride0,
      off1, stride1,
      off2, stride2
    ) => fn(
      i,
      new DenseVector(src0.data, off0, stride0, src0.rows),
      new DenseVector(src1.data, off1, stride1, src1.rows),
      new DenseVector(src2.data, off2, stride2, src2.rows)
    )
  )

  @inline
  final def foreachColumnVectorPair[T, U, V, W](src0: DenseMatrix[T],
                                                src1: DenseMatrix[U],
                                                src2: DenseMatrix[V],
                                                src3: DenseMatrix[W])
                                               (fn: (Int, DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W]) => Unit)
  : Unit = foreachColumnPair(src0, src1, src2, src3)(
    (
      i,
      off0, stride0,
      off1, stride1,
      off2, stride2,
      off3, stride3
    ) => fn(
      i,
      new DenseVector(src0.data, off0, stride0, src0.rows),
      new DenseVector(src1.data, off1, stride1, src1.rows),
      new DenseVector(src2.data, off2, stride2, src2.rows),
      new DenseVector(src3.data, off3, stride3, src3.rows)
    )
  )

  /*
  @inline
  final def foreachColumnVectorParallel[T](matrix0: DenseMatrix[T])
                                          (fn: DenseVector[T] => Unit)
  : Unit = {
    val tasks = mapColumnVectors(
      matrix0
    )(v0 => future(fn(v0)))
    ArrayEx.await(tasks)
  }

  @inline
  final def foreachColumnVectorParallel[T, U](matrix0: DenseMatrix[T],
                                              matrix1: DenseMatrix[U])
                                             (fn: (DenseVector[T], DenseVector[U]) => Unit)
  : Unit = {
    val tasks = zipColumnVectors(
      matrix0,
      matrix1
    )((v0, v1) => future(fn(v0, v1)))
    ArrayEx.await(tasks)
  }

  @inline
  final def foreachColumnVectorParallel[T, U, V](matrix0: DenseMatrix[T],
                                                 matrix1: DenseMatrix[U],
                                                 matrix2: DenseMatrix[V])
                                                (fn: (DenseVector[T], DenseVector[U], DenseVector[V]) => Unit)
  : Unit = {
    val tasks = zipColumnVectors(
      matrix0,
      matrix1,
      matrix2
    )((v0, v1, v2) => future(fn(v0, v1, v2)))
    ArrayEx.await(tasks)
  }

  @inline
  final def foreachColumnVectorParallel[T, U, V, W](matrix0: DenseMatrix[T],
                                                    matrix1: DenseMatrix[U],
                                                    matrix2: DenseMatrix[V],
                                                    matrix3: DenseMatrix[W])
                                                   (fn: (DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W]) => Unit)
  : Unit = {
    val tasks = zipColumnVectors(
      matrix0,
      matrix1,
      matrix2,
      matrix3
    )((v0, v1, v2, v3) => future(fn(v0, v1, v2, v3)))
    ArrayEx.await(tasks)
  }

  @inline
  final def foreachColumnVectorPairParallel[T](matrix0: DenseMatrix[T])
                                              (fn: (Int, DenseVector[T]) => Unit)
  : Unit = {
    val tasks = mapColumnVectorPairs(
      matrix0
    )((i, v0) => future(fn(i, v0)))
    ArrayEx.await(tasks)
  }

  @inline
  final def foreachColumnVectorPairParallel[T, U](matrix0: DenseMatrix[T],
                                                  matrix1: DenseMatrix[U])
                                                 (fn: (Int, DenseVector[T], DenseVector[U]) => Unit)
  : Unit = {
    val tasks = zipColumnVectorPairs(
      matrix0,
      matrix1
    )((i, v0, v1) => future(fn(i, v0, v1)))
    ArrayEx.await(tasks)
  }

  @inline
  final def foreachColumnVectorPairParallel[T, U, V](matrix0: DenseMatrix[T],
                                                     matrix1: DenseMatrix[U],
                                                     matrix2: DenseMatrix[V])
                                                    (fn: (Int, DenseVector[T], DenseVector[U], DenseVector[V]) => Unit)
  : Unit = {
    val tasks = zipColumnVectorPairs(
      matrix0,
      matrix1,
      matrix2
    )((i, v0, v1, v2) => future(fn(i, v0, v1, v2)))
    ArrayEx.await(tasks)
  }

  @inline
  final def foreachColumnVectorPairParallel[T, U, V, W](matrix0: DenseMatrix[T],
                                                        matrix1: DenseMatrix[U],
                                                        matrix2: DenseMatrix[V],
                                                        matrix3: DenseMatrix[W])
                                                       (fn: (Int, DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W]) => Unit)
  : Unit = {
    val tasks = zipColumnVectorPairs(
      matrix0,
      matrix1,
      matrix2,
      matrix3
    )((i, v0, v1, v2, v3) => future(fn(i, v0, v1, v2, v3)))
    ArrayEx.await(tasks)
  }
  */

  /*
  @inline
  final def foreachMinor[T](matrix0: DenseMatrix[T])
                           (fn: (Int, Int) => Unit)
  : Unit = {
    val stride0 = matrix0.majorStride
    val off0    = matrix0.offset
    val end0    = matrix0.offset + minor(matrix0)
    RangeEx.foreach(
      off0, end0,
      fn(_, stride0)
    )
  }

  @inline
  final def foreachMinorPair[T](matrix0: DenseMatrix[T])
                               (fn: (Int, Int, Int) => Unit)
  : Unit = {
    val stride0 = matrix0.majorStride
    val min0    = minor(matrix0)
    val off0    = matrix0.offset
    val end0    = matrix0.offset + min0
    RangeEx.foreachPair(
      off0, end0,
      fn(
        _,
        _, stride0,
        min0
      )
    )
  }
  */

  @inline
  final def foreachMajor[T](src0: DenseMatrix[T])
                           (fn: Int => Unit)
  : Unit = {
    val stride0 = src0.majorStride
    val off0    = src0.offset
    val end0    = src0.offset + stride0 * major(src0)
    RangeEx.foreach(
      off0, end0, stride0,
      fn
    )
  }

  @inline
  final def foreachMajor[T, U](src0: DenseMatrix[T],
                               src1: DenseMatrix[U])
                              (fn: (Int, Int) => Unit)
  : Unit = {
    val stride1 = src1.majorStride
    val maj1    = major(src1)
    var off1    = src1.offset

    val stride0 = src0.majorStride
    val maj0    = major(src0)
    var off0    = src0.offset
    val end0    = src0.offset + stride0 * maj0

    require(maj0 == maj1)

    while (off0 < end0) {
      fn(
        off0,
        off1
      )
      off1 += stride1
      off0 += stride0
    }
  }

  @inline
  final def foreachMajor[T, U, V](src0: DenseMatrix[T],
                                  src1: DenseMatrix[U],
                                  src2: DenseMatrix[V])
                                 (fn: (Int, Int, Int) => Unit)
  : Unit = {
    val stride2 = src2.majorStride
    val maj2    = major(src2)
    var off2    = src2.offset

    val stride1 = src1.majorStride
    val maj1    = major(src1)
    var off1    = src1.offset

    val stride0 = src0.majorStride
    val maj0    = major(src0)
    var off0    = src0.offset
    val end0    = src0.offset + stride0 * maj0

    require(
      maj0 == maj1 &&
      maj0 == maj2
    )

    while (off0 < end0) {
      fn(
        off0,
        off1,
        off2
      )
      off2 += stride2
      off1 += stride1
      off0 += stride0
    }
  }

  @inline
  final def foreachMajor[T, U, V, W](src0: DenseMatrix[T],
                                     src1: DenseMatrix[U],
                                     src2: DenseMatrix[V],
                                     src3: DenseMatrix[W])
                                    (fn: (Int, Int, Int, Int) => Unit)
  : Unit = {
    val stride3 = src3.majorStride
    val maj3    = major(src3)
    var off3    = src3.offset

    val stride2 = src2.majorStride
    val maj2    = major(src2)
    var off2    = src2.offset

    val stride1 = src1.majorStride
    val maj1    = major(src1)
    var off1    = src1.offset

    val stride0 = src0.majorStride
    val maj0    = major(src0)
    var off0    = src0.offset
    val end0    = src0.offset + stride0 * maj0

    require(
      maj0 == maj1 &&
      maj0 == maj2 &&
      maj0 == maj3
    )

    while (off0 < end0) {
      fn(
        off0,
        off1,
        off2,
        off3
      )
      off3 += stride3
      off2 += stride2
      off1 += stride1
      off0 += stride0
    }
  }

  @inline
  final def foreachMajorPair[T](src0: DenseMatrix[T])
                               (fn: (Int, Int) => Unit)
  : Unit = {
    val stride0 = src0.majorStride
    val off0    = src0.offset
    val end0    = src0.offset + stride0 * major(src0)
    RangeEx.foreachPair(
      off0, end0, stride0,
      fn
    )
  }

  @inline
  final def foreachEx[T, U](src0: DenseMatrix[T],
                            src1: Matrix[U])
                           (fn0: (T, U) => Unit, fn1: T => Unit)
  : Unit = src1 match {
    case src1: DenseMatrix[U] =>
      foreach(
        src0,
        src1
      )(fn0)
    case src1: CSCMatrix[U] =>
      foreachEx(
        src0,
        src1
      )(fn0, fn1)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def foreachEx[T, U, V](src0: DenseMatrix[T],
                               src1: DenseMatrix[U],
                               src2: CSCMatrix[V])
                              (fn0: (T, U, V) => Unit, fn1: (T, U) => Unit)
  : Unit = {
    // Precompute some values.
    val iter2     = src2.activeIterator

    val data1     = src1.data
    val rows1     = src1.rows
    val cols1     = src1.cols
    val stride1   = src1.majorStride
    val gap1      = stride1 - rows1
    var off1      = src1.offset

    val data0     = src0.data
    val rows0     = src0.rows
    val cols0     = src0.cols
    val stride0   = src0.majorStride
    val gap0      = stride0 - rows0
    var off0      = src0.offset

    require(
      !src0.isTranspose &&
      rows0 == src1.rows &&
      cols0 == src1.cols &&
      rows0 == src2.rows &&
      cols0 == src2.cols
    )

    // Process all pairs.
    var nextGap0 = off0 + rows0
    var r        = 0
    var c        = 0
    while (iter2.hasNext) {
      val next = iter2.next()
      while (c < next._1._2) {
        while (off0 < nextGap0) {
          fn1(data0(off0), data1(off1))
          off1 += 1
          off0 += 1
        }
        off1     += gap1
        off0     += gap0
        nextGap0  = off0 + rows0
        r         = 0
        c        += 1
      }
      while (r < next._1._1) {
        fn1(data0(off0), data1(off1))
        off1 += 1
        off0 += 1
        r    += 1
      }
      fn0(data0(off0), data1(off1), next._2)
      off1 += 1
      off0 += 1
      r    += 1
    }

    // If values remaining process them.
    val end0 = src0.offset + cols0 * stride0
    while (off0 < end0) {
      while (off0 < nextGap0) {
        fn1(data0(off0), data1(off1))
        off1 += 1
        off0 += 1
      }
      off1     += gap1
      off0     += gap0
      nextGap0  = off0 + rows0
    }
  }

  @inline
  final def foreachEx[T, U](src0: DenseMatrix[T],
                            src1: CSCMatrix[U])
                           (fn0: (T, U) => Unit, fn1: T => Unit)
  : Unit = {
    // Precompute some values.
    val iter1     = src1.activeIterator
    val data0     = src0.data
    val rows0     = src0.rows
    val cols0     = src0.cols
    val stride0   = src0.majorStride
    val gap0      = stride0 - rows0
    var off0      = src0.offset

    require(
      !src0.isTranspose &&
        rows0 == src1.rows &&
        cols0 == src1.cols
    )

    // Process all pairs.
    var nextGap0 = off0 + rows0
    var r        = 0
    var c        = 0
    while (iter1.hasNext) {
      val next = iter1.next()
      while (c < next._1._2) {
        while (off0 < nextGap0) {
          fn1(data0(off0))
          off0 += 1
        }
        off0  += gap0
        nextGap0  = off0 + rows0
        r         = 0
        c        += 1
      }
      while (r < next._1._1) {
        fn1(data0(off0))
        off0 += 1
        r    += 1
      }
      fn0(data0(off0), next._2)
      off0 += 1
      r    += 1
    }

    // If values remaining process them.
    val end0 = src0.offset + cols0 * stride0
    while (off0 < end0) {
      while (off0 < nextGap0) {
        fn1(data0(off0))
        off0 += 1
      }
      off0 += gap0
      nextGap0 = off0 + rows0
    }
  }

  @inline
  final def foreachPair[T](src0: DenseMatrix[T])
                          (fn: (Int, Int, T) => Unit)
  : Unit = {
    val data0   = src0.data
    val stride0 = src0.majorStride
    var off0    = src0.offset
    val rows0   = src0.rows
    val cols0   = src0.cols

    if (src0.isTranspose) {
      var r = 0
      while (r < rows0) {
        var c = 0
        while (c < cols0) {
          fn(r, c, data0(off0 + c))
          c += 1
        }
        off0 += stride0
        r    += 1
      }
    }
    else {
      var c = 0
      while (c < cols0) {
        var r = 0
        while (r < rows0) {
          fn(r, c, data0(off0 + r))
          r += 1
        }
        off0 += stride0
        c    += 1
      }
    }
  }

  @inline
  final def foreachRow[T](src0: DenseMatrix[T])
                         (fn: (Int, Int) => Unit)
  : Unit = {
    val stride0 = src0.majorStride
    var off0    = src0.offset

    if (src0.isTranspose) {
      val end0 = src0.offset + src0.rows * stride0
      while (off0 < end0) {
        fn(off0, 1)
        off0 += stride0
      }
    }
    else {
      val end0 = src0.offset + src0.rows
      while (off0 < end0) {
        fn(off0, stride0)
        off0 += 1
      }
    }
  }

  @inline
  final def foreachRow[T, U](src0: DenseMatrix[T],
                             src1: DenseMatrix[U])
                            (fn: (Int, Int, Int, Int) => Unit)
  : Unit = {
    require(
      src0.isTranspose == src1.isTranspose &&
      src0.rows        == src1.rows
    )

    val stride1 = src1.majorStride
    var off1    = src1.offset
    val stride0 = src0.majorStride
    var off0    = src0.offset

    if (src0.isTranspose) {
      val end0 = src0.offset + src0.rows * stride0
      while (off0 < end0) {
        fn(
          off0, 1,
          off1, 1
        )
        off1 += stride1
        off0 += stride0
      }
    }
    else {
      val end0 = src0.offset + src0.rows
      while (off0 < end0) {
        fn(
          off0, stride0,
          off1, stride1
        )
        off1 += 1
        off0 += 1
      }
    }
  }

  @inline
  final def foreachRow[T, U, V](src0: DenseMatrix[T],
                                src1: DenseMatrix[U],
                                src2: DenseMatrix[V])
                               (fn: (Int, Int, Int, Int, Int, Int) => Unit)
  : Unit = {
    require(
      src0.isTranspose == src1.isTranspose &&
      src0.rows        == src1.rows        &&
      src0.isTranspose == src2.isTranspose &&
      src0.rows        == src2.rows
    )

    val stride2 = src2.majorStride
    var off2    = src2.offset
    val stride1 = src1.majorStride
    var off1    = src1.offset
    val stride0 = src0.majorStride
    var off0    = src0.offset

    if (src0.isTranspose) {
      val end0 = src0.offset + src0.rows * stride0
      while (off0 < end0) {
        fn(
          off0, 1,
          off1, 1,
          off2, 1
        )
        off2 += stride2
        off1 += stride1
        off0 += stride0
      }
    }
    else {
      val end0 = src0.offset + src0.rows
      while (off0 < end0) {
        fn(
          off0, stride0,
          off1, stride1,
          off2, stride2
        )
        off2 += 1
        off1 += 1
        off0 += 1
      }
    }
  }

  @inline
  final def foreachRow[T, U, V, W](src0: DenseMatrix[T],
                                   src1: DenseMatrix[U],
                                   src2: DenseMatrix[V],
                                   src3: DenseMatrix[W])
                                  (fn: (Int, Int, Int, Int, Int, Int, Int, Int) => Unit)
  : Unit = {
    require(
      src0.isTranspose == src1.isTranspose &&
      src0.rows        == src1.rows        &&
      src0.isTranspose == src2.isTranspose &&
      src0.rows        == src2.rows        &&
      src0.isTranspose == src3.isTranspose &&
      src0.rows        == src3.rows
    )

    val stride3 = src3.majorStride
    var off3    = src3.offset
    val stride2 = src2.majorStride
    var off2    = src2.offset
    val stride1 = src1.majorStride
    var off1    = src1.offset
    val stride0 = src0.majorStride
    var off0    = src0.offset

    if (src0.isTranspose) {
      val end0 = src0.offset + src0.rows * stride0
      while (off0 < end0) {
        fn(
          off0, 1,
          off1, 1,
          off2, 1,
          off3, 1
        )
        off3 += stride3
        off2 += stride2
        off1 += stride1
        off0 += stride0
      }
    }
    else {
      val end0 = src0.offset + src0.rows
      while (off0 < end0) {
        fn(
          off0, stride0,
          off1, stride1,
          off2, stride2,
          off3, stride3
        )
        off3 += 1
        off2 += 1
        off1 += 1
        off0 += 1
      }
    }
  }

  @inline
  final def foreachRowPair[T](src0: DenseMatrix[T])
                             (fn: (Int, Int, Int) => Unit)
  : Unit = {
    val stride0 = src0.majorStride
    var off0    = src0.offset
    var i       = 0
    val n       = src0.rows

    if (src0.isTranspose) {
      while (i < n) {
        fn(
          i,
          off0, 1
        )
        off0 += stride0
        i    += 1
      }
    }
    else {
      while (i < n) {
        fn(
          i,
          off0, stride0
        )
        off0 += 1
        i    += 1
      }
    }
  }

  @inline
  final def foreachRowPair[T, U](src0: DenseMatrix[T],
                                 src1: DenseMatrix[U])
                                (fn: (Int, Int, Int, Int, Int) => Unit)
  : Unit = {
    require(
      src0.isTranspose == src1.isTranspose &&
      src0.rows        == src1.rows
    )

    val stride1 = src1.majorStride
    var off1    = src1.offset
    val stride0 = src0.majorStride
    var off0    = src0.offset
    var i       = 0
    val n       = src0.rows

    if (src0.isTranspose) {
      while (i < n) {
        fn(
          i,
          off0, 1,
          off1, 1
        )
        off1 += stride1
        off0 += stride0
        i    += 1
      }
    }
    else {
      while (i < n) {
        fn(
          i,
          off0, stride0,
          off1, stride1
        )
        off1 += 1
        off0 += 1
        i    += 1
      }
    }
  }

  @inline
  final def foreachRowPair[T, U, V](src0: DenseMatrix[T],
                                    src1: DenseMatrix[U],
                                    src2: DenseMatrix[V])
                                   (fn: (Int, Int, Int, Int, Int, Int, Int) => Unit)
  : Unit = {
    require(
      src0.isTranspose == src1.isTranspose &&
      src0.rows        == src1.rows        &&
      src0.isTranspose == src2.isTranspose &&
      src0.rows        == src2.rows
    )

    val stride2 = src2.majorStride
    var off2    = src2.offset
    val stride1 = src1.majorStride
    var off1    = src1.offset
    val stride0 = src0.majorStride
    var off0    = src0.offset
    var i       = 0
    val n       = src0.rows

    if (src0.isTranspose) {
      while (i < n) {
        fn(
          i,
          off0, 1,
          off1, 1,
          off2, 1
        )
        off2 += stride2
        off1 += stride1
        off0 += stride0
        i    += 1
      }
    }
    else {
      while (i < n) {
        fn(
          i,
          off0, stride0,
          off1, stride1,
          off2, stride2
        )
        off2 += 1
        off1 += 1
        off0 += 1
        i    += 1
      }
    }
  }

  @inline
  final def foreachRowPair[T, U, V, W](src0: DenseMatrix[T],
                                       src1: DenseMatrix[U],
                                       src2: DenseMatrix[V],
                                       src3: DenseMatrix[W])
                                      (fn: (Int, Int, Int, Int, Int, Int, Int, Int, Int) => Unit)
  : Unit = {
    require(
      src0.isTranspose == src1.isTranspose &&
      src0.rows        == src1.rows        &&
      src0.isTranspose == src2.isTranspose &&
      src0.rows        == src2.rows        &&
      src0.isTranspose == src3.isTranspose &&
      src0.rows        == src3.rows
    )

    val stride3 = src3.majorStride
    var off3    = src3.offset
    val stride2 = src2.majorStride
    var off2    = src2.offset
    val stride1 = src1.majorStride
    var off1    = src1.offset
    val stride0 = src0.majorStride
    var off0    = src0.offset
    var i       = 0
    val n       = src0.rows

    if (src0.isTranspose) {
      while (i < n) {
        fn(
          i,
          off0, 1,
          off1, 1,
          off2, 1,
          off3, 1
        )
        off3 += stride3
        off2 += stride2
        off1 += stride1
        off0 += stride0
        i    += 1
      }
    }
    else {
      while (i < n) {
        fn(
          i,
          off0, stride0,
          off1, stride1,
          off2, stride2,
          off3, stride3
        )
        off3 += 1
        off2 += 1
        off1 += 1
        off0 += 1
        i    += 1
      }
    }
  }

  @inline
  final def foreachRowVector[T](src0: DenseMatrix[T])
                               (fn: DenseVector[T] => Unit)
  : Unit = foreachRow(src0)(
    (off0, stride0) => fn(
      new DenseVector(src0.data, off0, stride0, src0.cols)
    )
  )

  @inline
  final def foreachRowVector[T, U](src0: DenseMatrix[T],
                                   src1: DenseMatrix[U])
                                  (fn: (DenseVector[T], DenseVector[U]) => Unit)
  : Unit = foreachRow(src0, src1)(
    (
      off0, stride0,
      off1, stride1
    ) => fn(
      new DenseVector(src0.data, off0, stride0, src0.cols),
      new DenseVector(src1.data, off1, stride1, src1.cols)
    )
  )

  @inline
  final def foreachRowVector[T, U, V](src0: DenseMatrix[T],
                                      src1: DenseMatrix[U],
                                      src2: DenseMatrix[V])
                                     (fn: (DenseVector[T], DenseVector[U], DenseVector[V]) => Unit)
  : Unit = foreachRow(src0, src1, src2)(
    (
      off0, stride0,
      off1, stride1,
      off2, stride2
    ) => fn(
      new DenseVector(src0.data, off0, stride0, src0.cols),
      new DenseVector(src1.data, off1, stride1, src1.cols),
      new DenseVector(src2.data, off2, stride2, src2.cols)
    )
  )

  @inline
  final def foreachRowVector[T, U, V, W](src0: DenseMatrix[T],
                                         src1: DenseMatrix[U],
                                         src2: DenseMatrix[V],
                                         src3: DenseMatrix[W])
                                        (fn: (DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W]) => Unit)
  : Unit = foreachRow(src0, src1, src2, src3)(
    (
      off0, stride0,
      off1, stride1,
      off2, stride2,
      off3, stride3
    ) => fn(
      new DenseVector(src0.data, off0, stride0, src0.cols),
      new DenseVector(src1.data, off1, stride1, src1.cols),
      new DenseVector(src2.data, off2, stride2, src2.cols),
      new DenseVector(src3.data, off3, stride3, src3.cols)
    )
  )

  @inline
  final def foreachRowVectorPair[T](src0: DenseMatrix[T])
                                   (fn: (Int, DenseVector[T]) => Unit)
  : Unit = foreachRowPair(src0)(
    (
      i,
      off0, stride0
    ) => fn(
      i,
      new DenseVector(src0.data, off0, stride0, src0.cols)
    )
  )

  @inline
  final def foreachRowVectorPair[T, U](src0: DenseMatrix[T],
                                       src1: DenseMatrix[U])
                                      (fn: (Int, DenseVector[T], DenseVector[U]) => Unit)
  : Unit = foreachRowPair(src0, src1)(
    (
      i,
      off0, stride0,
      off1, stride1
    ) => fn(
      i,
      new DenseVector(src0.data, off0, stride0, src0.cols),
      new DenseVector(src1.data, off1, stride1, src1.cols)
    )
  )

  @inline
  final def foreachRowVectorPair[T, U, V](src0: DenseMatrix[T],
                                          src1: DenseMatrix[U],
                                          src2: DenseMatrix[V])
                                         (fn: (Int, DenseVector[T], DenseVector[U], DenseVector[V]) => Unit)
  : Unit = foreachRowPair(src0, src1, src2)(
    (
      i,
      off0, stride0,
      off1, stride1,
      off2, stride2
    ) => fn(
      i,
      new DenseVector(src0.data, off0, stride0, src0.cols),
      new DenseVector(src1.data, off1, stride1, src1.cols),
      new DenseVector(src2.data, off2, stride2, src2.cols)
    )
  )

  @inline
  final def foreachRowVectorPair[T, U, V, W](src0: DenseMatrix[T],
                                             src1: DenseMatrix[U],
                                             src2: DenseMatrix[V],
                                             src3: DenseMatrix[W])
                                            (fn: (Int, DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W]) => Unit)
  : Unit = foreachRowPair(src0, src1, src2, src3)(
    (
      i,
      off0, stride0,
      off1, stride1,
      off2, stride2,
      off3, stride3
    ) => fn(
      i,
      new DenseVector(src0.data, off0, stride0, src0.cols),
      new DenseVector(src1.data, off1, stride1, src1.cols),
      new DenseVector(src2.data, off2, stride2, src2.cols),
      new DenseVector(src3.data, off3, stride3, src3.cols)
    )
  )

  /**
    * Vertical concatenation (values, interleaving).
    */
  @inline
  final def interleaveColumns[T](dst0: DenseMatrix[T],
                                 src1: DenseMatrix[T],
                                 src2: DenseMatrix[T])
                                (implicit tagT: ClassTag[T])
  : Unit = foreachColumn(dst0, src1, src2)(
    (
      off0, stride0,
      off1, stride1,
      off2, stride2
    ) => {
      ArrayEx.interleave(
        dst0.data, off0, stride0,
        src1.data, off1, stride1,
        src1.rows,
        src2.data, off2, stride2,
        src2.rows
      )
    }
  )

  /**
    * Vertical concatenation (values, interleaving).
    */
  @inline
  final def interleaveColumns[T](src0: DenseMatrix[T],
                                 src1: CSCMatrix[T])
                                (implicit tagT: ClassTag[T])
  : Array[T] = {
    // Create shorthands for frequently used variables.
    val rows1 = src1.rows
    val rows0 = src0.rows
    val cols0 = src0.cols
    val data0 = src0.data

    require(
      !src0.isTranspose &&
      rows0 == rows1       &&
      cols0 == src1.cols
    )

    // Allocate result buffer.
    val rowsR = rows0 + rows1
    val result = new Array[T](rowsR * cols0)

    // Fill result vector and return.
    var offR = 0
    foreachColumn(src0)((off0, stride0) => {
      ArrayEx.set(
        result, offR, 2,
        data0,  off0, stride0,
        rows0
      )
      offR = rowsR
    })

    foreachActivePair(src1)((r, c, v1) => {
      result(rows0 + c * rowsR + r + r + 1) = v1
    })

    result
  }

  /**
    * Vertical concatenation (values, interleaving).
    */
  @inline
  final def interleaveColumns[T](src0: CSCMatrix[T],
                                 src1: DenseMatrix[T])
                                (implicit classTag: ClassTag[T])
  : Array[T] = {
    // Create shorthands for frequently used variables.
    val rows1 = src1.rows
    val rows0 = src0.rows
    val cols0 = src0.cols
    val data0 = src0.data

    require(
      !src1.isTranspose &&
      rows0 == rows1       &&
      cols0 == src1.cols
    )

    // Allocate result buffer.
    val rowsR  = rows0 + rows1
    val result = new Array[T](rowsR * cols0)

    // Fill result vector and return.
    foreachActivePair(src0)((r, c, v1) => {
      result(rows0 + c * rowsR + r + r) = v1
    })

    var offR = rows0
    foreachColumn(src1)((off0, stride0) => {
      ArrayEx.set(
        result, offR, 2,
        data0,  off0, stride0,
        rows0
      )
      offR = rowsR
    })

    result
  }

  /**
    * Vertical concatenation (values, interleaving).
    */
  @inline
  final def interleaveColumns[T](src0: CSCMatrix[T],
                                 src1: CSCMatrix[T])
                                (implicit classTag: ClassTag[T], zero: Zero[T])
  : CSCMatrix[T] = {
    // Create shorthands for frequently used variables.
    val offsets1 = src1.colPtrs
    val indices1 = src1.rowIndices
    val data1    = src1.data
    val used1    = src1.activeSize
    val rows1    = src1.rows
    val offsets0 = src0.colPtrs
    val indices0 = src0.rowIndices
    val data0    = src0.data
    val used0    = src0.activeSize
    val rows0    = src0.rows
    val cols0    = src0.cols

    require(cols0 == src1.cols)

    // Allocate result buffers.
    val rowsR    = rows0 + rows1
    val usedR    = used0 + used1
    val offsetsR = new Array[Int](offsets0.length)
    val indicesR = new Array[Int](usedR)
    val dataR    = new Array[T](usedR)

    // Fill buffers.
    var off1 = offsets1(0)
    var off0 = offsets0(0)
    var offR = 0
    var c    = 1
    while (c < offsetsR.length) {
      val end0 = offsets0(c)
      val end1 = offsets1(c)

      // Fill up result buffer until either this or other is finished.
      while (off0 < end0 && off1 < end1) {
        val index0 = indices0(off0)
        val index1 = indices1(off1)

        if (index0 <= index1) {
          indicesR(offR) = index0 + index0
          dataR(offR)    = data0(off0)
          off0 += 1
          offR += 1
        }

        if (index0 >= index1) {
          indicesR(offR) = index1 + index1 + 1
          dataR(offR)    = data1(off1)
          off1 += 1
          offR += 1
        }
      }

      // Remaining items in dv.
      while (off0 < end0) {
        val index0     = indices0(off0)
        indicesR(offR) = index0 + index0
        dataR(offR)    = data0(off0)
        offR += 1
        off0 += 1
      }

      // Remaining items in other.
      while (off1 < end1) {
        val index1     = indices1(off1)
        indicesR(offR) = index1 + index1 + 1
        dataR(offR)    = data1(off1)
        offR += 1
        off1 += 1
      }

      // Set next col pointer.
      offsetsR(c) = offR
      c += 1
    }

    new CSCMatrix(dataR, rowsR, cols0, offsetsR, indicesR)
  }

  @inline
  final def head[T](src0: DenseMatrix[T])
  : T = {
    require(nonEmpty(src0))
    src0.data(src0.offset)
  }

  @inline
  final def l2NormSq(src0: DenseMatrix[Real])
  : Real = dot(src0, src0)

  @inline
  final def labelsToDense(noClasses: Int,
                          classNo:   Int)
  : DenseMatrix[Real] = labelsToDense(noClasses, classNo, Real.one)

  @inline
  final def labelsToDense(noClasses: Int,
                          classNo:   Int,
                          value:     Real)
  : DenseMatrix[Real] = new DenseMatrix(
    noClasses, 1, ArrayEx.labelsToArray(noClasses, classNo, value)
  )

  @inline
  final def labelsToDense(noClasses: Int,
                          classNo:   Seq[Int])
  : DenseMatrix[Real] = labelsToDense(noClasses, classNo, Real.one)

  @inline
  final def labelsToDense(noClasses: Int,
                          classNo:   Seq[Int],
                          value:     Real)
  : DenseMatrix[Real] = {
    require(classNo.forall(i => i >= 0 && i < noClasses))
    val result = DenseMatrix.zeros[Real](noClasses, classNo.length)
    SeqEx.foreachPair(classNo)(
      (c, r) => result.update(r, c, value)
    )
    result
  }

  @inline
  final def labelsToDenseEx(noClasses: Int,
                            classNo:   Seq[Seq[Int]])
  : DenseMatrix[Real] = labelsToDenseEx(noClasses, classNo, Real.one)

  @inline
  final def labelsToDenseEx(noClasses: Int,
                            classNo:   Seq[Seq[Int]],
                            value:     Real)
  : DenseMatrix[Real] = {
    // TODO: Error checking!
    val result = DenseMatrix.zeros[Real](noClasses, classNo.length)
    SeqEx.foreachPair(
      classNo
    )(
      (c, classNo) => classNo.foreach(
        r => result.update(r, c, value)
      )
    )
    result
  }

  @inline
  final def labelsToSparse(noClasses: Int,
                           classNo:   Int)
  : CSCMatrix[Real] = labelsToSparse(noClasses, classNo, Real.one)

  @inline
  final def labelsToSparse(noClasses: Int,
                           classNo:   Int,
                           value:     Real)
  : CSCMatrix[Real] = {
    require(classNo >= 0 && classNo < noClasses)
    val offsets = Array(0, 1)
    val indices = Array(classNo)
    val data    = Array(value)
    new CSCMatrix(
      data,
      noClasses,
      1,
      offsets,
      indices
    )
  }

  @inline
  final def labelsToSparse(noClasses: Int,
                           classNo:   Seq[Int])
  : CSCMatrix[Real] = labelsToSparse(noClasses, classNo, Real.one)

  @inline
  final def labelsToSparse(noClasses: Int,
                           classNo:   Seq[Int],
                           value:     Real)
  : CSCMatrix[Real] = {
    require(classNo.forall(i => i >= 0 && i < noClasses))
    val used    = classNo.length
    val offsets = ArrayEx.tabulate[Int](used + 1)(i => i)
    val indices = classNo.toArray
    val data    = ArrayEx.fill(used, value)
    new CSCMatrix(
      data,
      noClasses,
      used,
      offsets,
      indices
    )
  }

  @inline
  final def labelsToSparseEx(noClasses: Int,
                             classNo:   Seq[Seq[Int]])
  : CSCMatrix[Real] = labelsToSparseEx(noClasses, classNo, Real.one)

  @inline
  final def labelsToSparseEx(noClasses: Int,
                             classNo:   Seq[Seq[Int]],
                             value:     Real)
  : CSCMatrix[Real] = {
    val result = CSCMatrix.zeros[Real](noClasses, classNo.length)
    SeqEx.foreachPair(classNo)((c, classNo) => classNo.foreach(
      r => result.update(r, c, value)
    ))
    result
  }

  @inline
  final def leftColumn[T](src0: DenseMatrix[T])
  : DenseVector[T] = src0(::, 0)

  @inline
  final def lerp(dst0: DenseMatrix[Real],
                 src1: DenseMatrix[Real],
                 t:    Real)
  : Unit = {
    val data1 = src1.data
    val data0 = dst0.data
    val rows0 = dst0.rows

    require(rows0 == src1.rows)

    // TODO: Could be done faster!
    foreachColumn(dst0, src1)(
      (
        off0, stride0,
        off1, stride1
      ) => {
        ArrayEx.lerp(
          data0, off0, stride0,
          data1, off1, stride1,
          rows0,
          t
        )
      }
    )
  }

  @inline
  final def map[T, U](src0: DenseMatrix[T])
                     (fn: T => U)
                     (implicit tagU: ClassTag[U])
  : Array[U] = {
    val data0   = src0.data
    val min0    = minor(src0)
    val gap0    = src0.majorStride - min0
    var offset0 = src0.offset

    val result = new Array[U](src0.size)
    var i      = 0
    while (i < result.length) {
      val nextGap = i + min0
      while (i < nextGap) {
        result(i) = fn(data0(offset0))
        offset0 += 1
        i       += 1
      }
      offset0 += gap0
    }
    result
  }

  @inline
  final def mapActive[T, U](src0: CSCMatrix[T])
                           (fn: T => U)
                           (implicit tagU: ClassTag[U], zeroU: Zero[U])
  : CSCMatrix[U] = {
    val offsetsR = src0.colPtrs.clone()
    val usedR    = offsetsR(src0.cols)
    val indicesR = ArrayEx.take(src0.rowIndices, usedR)
    val dataR    = ArrayEx.map(
      src0.data, 0, 1,
      usedR
    )(fn)
    new CSCMatrix(
      dataR,
      src0.rows,
      src0.cols,
      offsetsR,
      indicesR
    )
  }

  @inline
  final def mapColumns[T, U](src0: DenseMatrix[T])
                            (fn: (Int, Int) => U)
                            (implicit tagU: ClassTag[U])
  : Array[U] = {
    val result = new Array[U](src0.cols)
    foreachColumnPair(
      src0
    )((i, off0, stride0) => result(i) = fn(off0, stride0))
    result
  }

  @inline
  final def mapColumnPairs[T, U](src0: DenseMatrix[T])
                                (fn: (Int, Int, Int) => U)
                                (implicit tagU: ClassTag[U])
  : Array[U] = {
    val result = new Array[U](src0.cols)
    foreachColumnPair(
      src0
    )((i, off0, stride0) => result(i) = fn(i, off0, stride0))
    result
  }

  @inline
  final def mapColumnVectors[T, U](src0: DenseMatrix[T])
                                  (fn: DenseVector[T] => U)
                                  (implicit tagU: ClassTag[U])
  : Array[U] = {
    val result = new Array[U](src0.cols)
    foreachColumnVectorPair(
      src0
    )((i, col) => result(i) = fn(col))
    result
  }

  @inline
  final def mapColumnVectorPairs[T, U](src0: DenseMatrix[T])
                                      (fn: (Int, DenseVector[T]) => U)
                                      (implicit tagU: ClassTag[U])
  : Array[U] = {
    val result = new Array[U](src0.cols)
    foreachColumnVectorPair(
      src0
    )((i, col) => result(i) = fn(i, col))
    result
  }

  /*
  @inline
  final def mapMajors[T, U](matrix0: DenseMatrix[T])
                           (fn: Int => U)
                           (implicit tagU: ClassTag[U])
  : Array[U] = {
    val result = new Array[U](major(matrix0))
    foreachMinorPair(
      matrix0
    )((i, off0) => result(i) = fn(off0))
    result
  }

  @inline
  final def mapMinorPairs[T, U](matrix0: DenseMatrix[T])
                               (fn: (Int, Int) => U)
                               (implicit tagU: ClassTag[U])
  : Array[U] = {
    val result = new Array[U](major(matrix0))
    foreachMinorPair(
      matrix0
    )((i, off0) => result(i) = fn(i, off0))
    result
  }

  @inline
  final def mapMinorVectors[T, U](matrix0: DenseMatrix[T])
                                 (fn: DenseVector[T] => U)
                                 (implicit tagU: ClassTag[U])
  : Array[U] = {
    val result = new Array[U](major(matrix0))
    foreachMinorVectorPair(
      matrix0
    )((i, v0) => result(i) = fn(v0))
    result
  }

  @inline
  final def mapMinorVectorPairs[T, U](matrix0: DenseMatrix[T])
                                     (fn: (Int, DenseVector[T]) => U)
                                     (implicit tagU: ClassTag[U])
  : Array[U] = {
    val result = new Array[U](major(matrix0))
    foreachMinorVectorPair(
      matrix0
    )((i, v0) => result(i) = fn(i, v0))
    result
  }
  */

  @inline
  final def mapRowVectors[T, U](src0: DenseMatrix[T])
                               (fn: DenseVector[T] => U)
                               (implicit tagU: ClassTag[U])
  : Array[U] = {
    val result = new Array[U](src0.rows)
    foreachRowVectorPair(
      src0
    )((i, row0) => result(i) = fn(row0))
    result
  }

  @inline
  final def mapRowVectorPairs[T, U](src0: DenseMatrix[T])
                                   (fn: (Int, DenseVector[T]) => U)
                                   (implicit tagU: ClassTag[U])
  : Array[U] = {
    val result = new Array[U](src0.rows)
    foreachRowVectorPair(
      src0
    )((i, row0) => result(i) = fn(i, row0))
    result
  }

  @inline
  final def mapRowVectorPairs[T, U, V](src0: DenseMatrix[T],
                                       src1: DenseMatrix[U])
                                      (fn: (Int, DenseVector[T], DenseVector[U]) => V)
                                      (implicit tagV: ClassTag[V])
  : Array[V] = {
    require(src0.rows == src1.rows)
    val result = new Array[V](src0.rows)
    foreachRowVectorPair(
      src0,
      src1
    )((i, row0, row1) => result(i) = fn(i, row0, row1))
    result
  }

  @inline
  final def max(src0: CSCMatrix[Real])
  : Real = {
    val tmp = ArrayEx.max(
      src0.data, 0, 1,
      src0.activeSize
    )
    if (src0.activeSize < src0.size) {
      MathMacros.max(tmp, Real.zero)
    }
    else {
      tmp
    }
  }

  @inline
  final def maxAbs(src0: CSCMatrix[Real])
  : Real = {
    if (src0.activeSize <= 0 && src0.size > 0) {
      Real.zero
    }
    else {
      ArrayEx.maxAbs(
        src0.data, 0, 1,
        src0.activeSize
      )
    }
  }

  @inline
  final def min(src0: CSCMatrix[Real])
  : Real = {
    val tmp = ArrayEx.min(
      src0.data, 0, 1,
      src0.activeSize
    )
    if (src0.activeSize < src0.size) {
      MathMacros.min(tmp, Real.zero)
    }
    else {
      tmp
    }
  }

  @inline
  final def minor[T](src0: DenseMatrix[T])
  : Int = if (src0.isTranspose) src0.cols else src0.rows

  /*
  @inline
  final def minors[T](matrix0: DenseMatrix[T])
  : Range = {
    val off0 = matrix0.offset
    val end0 = matrix0.offset + minor(matrix0)
    Range(off0, end0, 1)
  }
  */

  /*
  @inline
  final def majorVectors[T](matrix0: DenseMatrix[T])
  : Array[DenseVector[T]] = {
    val data0   = matrix0.data
    val maj0    = major(matrix0)
    val result  = new Array[DenseVector[T]](minor(matrix0))
    foreachMajorPair(
      matrix0
    )((i, off0, stride0) => result(i) = new DenseVector(data0, off0, stride0, maj0))
    result
  }
  */

  @inline
  final def major[T](src0: DenseMatrix[T])
  : Int = if (src0.isTranspose) src0.rows else src0.cols

  @inline
  final def majors[T](src0: DenseMatrix[T])
  : Range = {
    val off0    = src0.offset
    val stride0 = src0.majorStride
    val maj0    = major(src0)
    val end0    = src0.offset + stride0 * maj0
    Range(off0, end0, stride0)
  }

  @inline
  final def maxIndex(src0: DenseMatrix[Real])
  : (Int, Int) = {
    val rows0    = src0.rows
    val data0    = src0.data
    var maxValue = data0(src0.offset)
    var maxRow   = 0
    var maxCol   = 0
    foreachColumnPair(src0)(
      (c, off0, stride0) => {
        val r = ArrayEx.maxIndex(
          data0, off0, stride0,
          rows0
        )
        val v = data0(off0 + r * stride0)
        if (v > maxValue) {
          maxValue = v
          maxRow   = r
          maxCol   = c
        }
      }
    )
    (maxRow, maxCol)
  }

  /*
  @inline
  final def minorsParallel[T](matrix0: DenseMatrix[T])
  : ParArray[Int] = ParArray.handoff(minors(matrix0))

  @inline
  final def minorsPairsParallel[T](matrix0: DenseMatrix[T])
  : ParArray[(Int, Int)] = ParArray.handoff(minorPairs(matrix0))
  */

  @inline
  final def minorVectors[T](src0: DenseMatrix[T])
  : Array[DenseVector[T]] = {
    val data0  = src0.data
    val min0   = minor(src0)
    val result = new Array[DenseVector[T]](major(src0))
    foreachMajorPair(src0)(
      (i, off0) => result(i) = new DenseVector(data0, off0, 1, min0)
    )
    result
  }

  @inline
  final def minorVectorPairs[T](src0: DenseMatrix[T])
  : Array[(Int, DenseVector[T])] = {
    val data0 = src0.data
    val min0  = minor(src0)
    val result = new Array[(Int, DenseVector[T])](major(src0))
    foreachMajorPair(src0)(
      (i, off0) => result(i) = (i, new DenseVector(data0, off0, 1, min0))
    )
    result
  }

  @inline
  final def nonEmpty[T](src0: DenseMatrix[T])
  : Boolean = src0.rows > 0 && src0.cols > 0

  @inline
  final def reshape[T](src0: DenseMatrix[T], rows: Int)
  : DenseMatrix[T] = reshape(src0, rows, src0.size / rows)

  @inline
  final def reshape[T](src0: DenseMatrix[T], rows: Int, cols: Int)
  : DenseMatrix[T] = src0.reshape(rows, cols, View.Require)

  @inline
  final def rightColumn[T](src0: DenseMatrix[T])
  : DenseVector[T] = src0(::, -1)

  @inline
  final def rowVector[T](src0: DenseMatrix[T], index: Int)
  : DenseVector[T] = {
    require(!src0.isTranspose && index >= 0 && index < src0.rows)
    new DenseVector(
      src0.data,
      src0.offset + index,
      src0.majorStride,
      src0.cols
    )
  }

  @inline
  final def rowVectors[T](src0: DenseMatrix[T])
  : Array[DenseVector[T]] = {
    val result = new Array[DenseVector[T]](
      src0.cols
    )
    foreachRowVectorPair(
      src0
    )((i, v0) => result(i) = v0)
    result
  }

  @inline
  final def rowVectors[T, U](src0: DenseMatrix[T],
                             src1: DenseMatrix[U])
  : Array[(DenseVector[T], DenseVector[U])] = {
    val result = new Array[(DenseVector[T], DenseVector[U])](
      src0.cols
    )
    foreachRowVectorPair(
      src0,
      src1
    )((i, v0, v1) => result(i) = (v0, v1))
    result
  }

  @inline
  final def rowVectors[T, U, V](src0: DenseMatrix[T],
                                src1: DenseMatrix[U],
                                src2: DenseMatrix[V])
  : Array[(DenseVector[T], DenseVector[U], DenseVector[V])] = {
    val result = new Array[(DenseVector[T], DenseVector[U], DenseVector[V])](
      src0.cols
    )
    foreachRowVectorPair(
      src0,
      src1,
      src2
    )((i, v0, v1, v2) => result(i) = (v0, v1, v2))
    result
  }

  @inline
  final def rowVectors[T, U, V, W](src0: DenseMatrix[T],
                                   src1: DenseMatrix[U],
                                   src2: DenseMatrix[V],
                                   src3: DenseMatrix[W])
  : Array[(DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W])] = {
    val result = new Array[(DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W])](
      src0.cols
    )
    foreachRowVectorPair(
      src0,
      src1,
      src2,
      src3
    )((i, v0, v1, v2, v3) => result(i) = (v0, v1, v2, v3))
    result
  }

  /*
  @inline
  final def rowVectorsParallel[T](matrix0: DenseMatrix[T])
  : ParArray[DenseVector[T]] = {
    ParArray.handoff(rowVectors(matrix0))
  }

  @inline
  final def rowVectorsParallel[T, U](matrix0: DenseMatrix[T],
                                        matrix1: DenseMatrix[U])
  : ParArray[(DenseVector[T], DenseVector[U])] = {
    ParArray.handoff(rowVectors(matrix0, matrix1))
  }

  @inline
  final def crowVectorsParallel[T, U, V](matrix0: DenseMatrix[T],
                                           matrix1: DenseMatrix[U],
                                           matrix2: DenseMatrix[V])
  : ParArray[(DenseVector[T], DenseVector[U], DenseVector[V])] = {
    ParArray.handoff(rowVectors(matrix0, matrix1, matrix2))
  }

  @inline
  final def rowVectorsParallel[T, U, V, W](matrix0: DenseMatrix[T],
                                              matrix1: DenseMatrix[U],
                                              matrix2: DenseMatrix[V],
                                              matrix3: DenseMatrix[W])
  : ParArray[(DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W])] = {
    ParArray.handoff(rowVectors(matrix0, matrix1, matrix2, matrix3))
  }
  */

  @inline
  final def rowVectorPairs[T](src0: DenseMatrix[T])
  : Array[(Int, DenseVector[T])] = {
    val result = new Array[(Int, DenseVector[T])](
      src0.cols
    )
    foreachRowVectorPair(
      src0
    )((i, v0) => result(i) = (i, v0))
    result
  }

  @inline
  final def rowVectorPairs[T, U](src0: DenseMatrix[T],
                                 src1: DenseMatrix[U])
  : Array[(Int, DenseVector[T], DenseVector[U])] = {
    val result = new Array[(Int, DenseVector[T], DenseVector[U])](
      src0.cols
    )
    foreachRowVectorPair(
      src0,
      src1
    )((i, v0, v1) => result(i) = (i, v0, v1))
    result
  }

  @inline
  final def rowVectorPairs[T, U, V](src0: DenseMatrix[T],
                                    src1: DenseMatrix[U],
                                    src2: DenseMatrix[V])
  : Array[(Int, DenseVector[T], DenseVector[U], DenseVector[V])] = {
    val result = new Array[(Int, DenseVector[T], DenseVector[U], DenseVector[V])](
      src0.cols
    )
    foreachRowVectorPair(
      src0,
      src1,
      src2
    )((i, v0, v1, v2) => result(i) = (i, v0, v1, v2))
    result
  }

  @inline
  final def rowVectorPairs[T, U, V, W](src0: DenseMatrix[T],
                                       src1: DenseMatrix[U],
                                       src2: DenseMatrix[V],
                                       src3: DenseMatrix[W])
  : Array[(Int, DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W])] = {
    val result = new Array[(Int, DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W])](
      src0.cols
    )
    foreachRowVectorPair(
      src0,
      src1,
      src2,
      src3
    )((i, v0, v1, v2, v3) => result(i) = (i, v0, v1, v2, v3))
    result
  }

  /*
  @inline
  final def rowVectorPairsParallel[T](matrix0: DenseMatrix[T])
  : ParArray[(Int, DenseVector[T])] = {
    ParArray.handoff(rowVectorPairs(matrix0))
  }

  @inline
  final def rowVectorPairsParallel[T, U](matrix0: DenseMatrix[T],
                                         matrix1: DenseMatrix[U])
  : ParArray[(Int, DenseVector[T], DenseVector[U])] = {
    ParArray.handoff(rowVectorPairs(matrix0, matrix1))
  }

  @inline
  final def rowVectorPairsParallel[T, U, V](matrix0: DenseMatrix[T],
                                            matrix1: DenseMatrix[U],
                                            matrix2: DenseMatrix[V])
  : ParArray[(Int, DenseVector[T], DenseVector[U], DenseVector[V])] = {
    ParArray.handoff(rowVectorPairs(matrix0, matrix1, matrix2))
  }

  @inline
  final def rowVectorPairsParallel[T, U, V, W](matrix0: DenseMatrix[T],
                                               matrix1: DenseMatrix[U],
                                               matrix2: DenseMatrix[V],
                                               matrix3: DenseMatrix[W])
  : ParArray[(Int, DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W])] = {
    ParArray.handoff(rowVectorPairs(matrix0, matrix1, matrix2, matrix3))
  }
  */

  @inline
  final def sqr(src0: CSCMatrix[Real])
  : Unit = ArrayEx.sqr(
    src0.data, 0, 1,
    src0.activeSize
  )

  @inline
  final def sqrt(src0: CSCMatrix[Real])
  : Unit = ArrayEx.sqrt(
    src0.data, 0, 1,
    src0.activeSize
  )

  @inline
  final def subtract(src0: Real,
                     src1: CSCMatrix[Real])
  : Unit = ArrayEx.subtract(
    src0,
    src1.data, 0, 1,
    src1.activeSize
  )

  @inline
  final def sum(src0: Matrix[Real])
  : Real = src0 match {
    case src0: DenseMatrix[Real] =>
      sum(src0)
    case src0: CSCMatrix[Real] =>
      sum(src0)
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def sum(src0: DenseMatrix[Real])
  : Real = {
    val data0 = src0.data
    val min0  = minor(src0)
    foldLeftMinors(
      Real.zero,
      src0
    )(_ + ArrayEx.sum(data0, _, 1, min0))
  }

  @inline
  final def sum(src0: CSCMatrix[Real])
  : Real = ArrayEx.sum(
    src0.data, 0, 1,
    src0.activeSize
  )

  @inline
  final def tabulate[T](dst0: DenseMatrix[T])
                       (fn: (Int, Int) => T)
  : Unit = {
    val data0   = dst0.data
    val stride0 = dst0.majorStride
    var offset0 = dst0.offset
    val rows0   = dst0.rows
    val cols0   = dst0.cols

    if (dst0.isTranspose) {
      var r = 0
      while (r < rows0) {
        var c = 0
        while (c < cols0) {
          data0(offset0 + c) = fn(r, c)
          c += 1
        }
        offset0 += stride0
        r       += 1
      }
    }
    else {
      var c = 0
      while (c < cols0) {
        var r = 0
        while (r < rows0) {
          data0(offset0 + r) = fn(r, c)
          r += 1
        }
        offset0 += stride0
        c       += 1
      }
    }
  }

  @inline
  final def toArray[T](src0: Matrix[T])
                      (implicit tagT: ClassTag[T])
  : Array[T] = src0 match {
    case src0: DenseMatrix[T] =>
      toArray(src0)
    case src0: CSCMatrix[T] =>
      toArray(src0)
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def toArray[T](src0: DenseMatrix[T])
                      (implicit tagT: ClassTag[T])
  : Array[T] = {
    val stride0 = src0.majorStride
    val maj0    = major(src0)
    val min0    = minor(src0)
    val result  = new Array[T](maj0 * min0)

    if (stride0 == min0) {
      ArrayEx.set(
        result,    0,           1,
        src0.data, src0.offset, 1,
        result.length
      )
    }
    else {
      var off1 = 0
      foreachMajor(src0)(off0 => {
        /*
        var i = 0
        while (i < maj0) {
          result(off1) = src0.data(off0 + 1)
          i += 1
        }*/

        ArrayEx.set(
          result,    off1, 1,
          src0.data, off0, 1,
          min0
        )
        off1 += min0
      })
      assume(off1 == result.length)
    }
    result
  }

  @inline
  final def toArray[T](src0: CSCMatrix[T])
                      (implicit tagT: ClassTag[T])
  : Array[T] = toArray(src0.toDense)

  @inline
  final def tailor[T](dst0: CSCMatrix[T])
                     (implicit tagT: ClassTag[T])
  : Unit = {
    val offsets = dst0.colPtrs
    val used    = offsets(dst0.cols)
    val data    = ArrayEx.take(dst0.data,       used)
    val indices = ArrayEx.take(dst0.rowIndices, used)
    dst0.use(data, offsets, indices, used)
  }

  @inline
  final def topRow[T](src0: DenseMatrix[T])
  : Transpose[DenseVector[T]] = src0(0, ::)

  /**
    * This is probably the most efficient single core method to update a dense
    * matrix. However, this works only well, if the virtual call to "fn" can
    * be stripped by the runtime. The next variant (below) seems to be more
    * reliable, but requires a true UFunc.
    *
    * @param fn Function to execute on each item.
    */
  @inline
  final def transform[T](dst0: DenseMatrix[T])
                        (fn: T => T)
  : Unit = {
    val data0   = dst0.data
    val min0    = minor(dst0)
    val stride0 = dst0.majorStride

    if (stride0 == min0) {
      ArrayEx.transform(
        data0, dst0.offset, 1,
        dst0.size
      )(fn)
    }
    else {
      foreachMajor(dst0)(
        ArrayEx.transform(
          data0, _, 1,
          min0
        )(fn)
      )
    }
  }

  /**
    * A more advanced version of updateEach. What this basically does is a
    * zip.inPlace.
    */
  @inline
  final def transform[T, U](dst0: DenseMatrix[T],
                            src1: DenseMatrix[U])
                           (fn: (T, U) => T)
  : Unit = {
    val data1   = src1.data
    val min1    = minor(src1)
    val stride1 = src1.majorStride

    val data0   = dst0.data
    val min0    = minor(dst0)
    val stride0 = dst0.majorStride

    require(
      dst0.isTranspose == src1.isTranspose &&
      dst0.rows        == src1.rows        &&
      dst0.cols        == src1.cols
    )

    if (
      min0 == stride0 &&
      min1 == stride1
    ) {
      ArrayEx.transform(
        data0, dst0.offset, 1,
        data1, src1.offset, 1,
        dst0.size
      )(fn)
    }
    else {
      foreachMajor(dst0, src1)(
        ArrayEx.transform(
          data0, _, 1,
          data1, _, 1,
          min0
        )(fn)
      )
    }
  }

  @inline
  final def transform[T, U, V](dst0: DenseMatrix[T],
                               src1: DenseMatrix[U],
                               src2: DenseMatrix[V])
                              (fn: (T, U, V) => T)
  : Unit = {
    val data2   = src2.data
    val min2    = minor(src2)
    val stride2 = src2.majorStride

    val data1   = src1.data
    val min1    = minor(src1)
    val stride1 = src1.majorStride

    val data0   = dst0.data
    val min0    = minor(dst0)
    val stride0 = dst0.majorStride

    require(
      dst0.isTranspose == src1.isTranspose &&
      dst0.rows        == src1.rows        &&
      dst0.cols        == src1.cols        &&
      dst0.isTranspose == src2.isTranspose &&
      dst0.rows        == src2.rows        &&
      dst0.cols        == src2.cols
    )

    if (
      min0 == stride0 &&
      min1 == stride1 &&
      min2 == stride2
    ) {
      ArrayEx.transform(
        data0, dst0.offset, 1,
        data1, src1.offset, 1,
        data2, src2.offset, 1,
        dst0.size
      )(fn)
    }
    else {
      foreachMajor(dst0, src1, src2)(
        ArrayEx.transform(
          data0, _, 1,
          data1, _, 1,
          data2, _, 1,
          min0
        )(fn)
      )
    }
  }

  @inline
  final def transform[T, U, V, W](dst0: DenseMatrix[T],
                                  src1: DenseMatrix[U],
                                  src2: DenseMatrix[V],
                                  src3: DenseMatrix[W])
                                 (fn: (T, U, V, W) => T)
  : Unit = {
    val data3   = src3.data
    val min3    = minor(src3)
    val stride3 = src3.majorStride

    val data2   = src2.data
    val min2    = minor(src2)
    val stride2 = src2.majorStride

    val data1   = src1.data
    val min1    = minor(src1)
    val stride1 = src1.majorStride

    val data0   = dst0.data
    val min0    = minor(dst0)
    val stride0 = dst0.majorStride

    require(
      dst0.isTranspose == src1.isTranspose &&
      dst0.rows        == src1.rows        &&
      dst0.cols        == src1.cols        &&
      dst0.isTranspose == src2.isTranspose &&
      dst0.rows        == src2.rows        &&
      dst0.cols        == src2.cols        &&
      dst0.isTranspose == src3.isTranspose &&
      dst0.rows        == src3.rows        &&
      dst0.cols        == src3.cols
    )

    if (
      min0 == stride0 &&
      min1 == stride1 &&
      min2 == stride2 &&
      min3 == stride3
    ) {
      ArrayEx.transform(
        data0, dst0.offset, 1,
        data1, src1.offset, 1,
        data2, src2.offset, 1,
        data3, src3.offset, 1,
        dst0.size
      )(fn)
    }
    else {
      foreachMajor(dst0, src1, src2, src3)(
        ArrayEx.transform(
          data0, _, 1,
          data1, _, 1,
          data2, _, 1,
          data3, _, 1,
          min0
        )(fn)
      )
    }
  }

  @inline
  final def transformActive[T](dst0: CSCMatrix[T])
                              (fn: T => T)
  : Unit = {
    val used = dst0.activeSize
    val data = dst0.data
    var i    = 0
    while (i < used) {
      data(i) = fn(data(i))
      i += 1
    }
  }

  @inline
  final def transformEx[T, U](dst0: DenseMatrix[T],
                              src1: Matrix[U])
                             (fn0: (T, U) => T, fn1: T => T)
  : Unit = src1 match {
    case src1: DenseMatrix[U] =>
      transform(
        dst0,
        src1
      )(fn0)
    case src1: CSCMatrix[U] =>
      transformEx(
        dst0,
        src1
      )(fn0, fn1)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def transformEx[T, U](dst0: DenseMatrix[T],
                              src1: CSCMatrix[U])
                             (fn0: (T, U) => T, fn1: T => T)
  : Unit = {
    // Precompute some values.
    val iter1   = src1.activeIterator

    val data0   = dst0.data
    val rows0   = dst0.rows
    val cols0   = dst0.cols
    val stride0 = dst0.majorStride
    val gap0    = stride0 - rows0
    var off0    = dst0.offset

    require(
      !dst0.isTranspose &&
      rows0 == src1.rows &&
      cols0 == src1.cols
    )

    // Process all pairs.
    var nextGap0 = off0 + rows0
    var r        = 0
    var c        = 0
    while (iter1.hasNext) {
      val next = iter1.next()
      while (c < next._1._2) {
        while (off0 < nextGap0) {
          data0(off0) = fn1(data0(off0))
          off0 += 1
        }
        off0     += gap0
        nextGap0  = off0 + rows0
        r         = 0
        c        += 1
      }
      while (r < next._1._1) {
        data0(off0) = fn1(data0(off0))
        off0 += 1
        r    += 1
      }
      data0(off0) = fn0(data0(off0), next._2)
      off0 += 1
      r    += 1
    }

    // If values remaining process them.
    val end0 = dst0.offset + cols0 * stride0
    while (off0 < end0) {
      while (off0 < nextGap0) {
        data0(off0) = fn1(data0(off0))
        off0 += 1
      }
      off0 += gap0
      nextGap0 = off0 + rows0
    }
  }

  @inline
  final def transformEx[T, U, V](dst0: DenseMatrix[T],
                                 src1: DenseMatrix[U],
                                 src2: Matrix[V])
                                (fn0: (T, U, V) => T, fn1: (T, U) => T)
  : Unit = src2 match {
    case src2: DenseMatrix[V] =>
      transform(
        dst0,
        src1,
        src2
      )(fn0)
    case src2: CSCMatrix[V] =>
      transformEx(
        dst0,
        src1,
        src2
      )(fn0, fn1)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def transformEx[T, U, V](dst0: DenseMatrix[T],
                                 src1: DenseMatrix[U],
                                 src2: CSCMatrix[V])
                                (fn0: (T, U, V) => T, fn1: (T, U) => T)
  : Unit = {
    // Precompute some values.
    val iter2   = src2.activeIterator

    val data1   = src1.data
    val stride1 = src1.majorStride
    val gap1    = stride1 - src1.rows
    var off1    = src1.offset

    val data0   = dst0.data
    val rows0   = dst0.rows
    val cols0   = dst0.cols
    val stride0 = dst0.majorStride
    val gap0    = stride0 - rows0
    var off0    = dst0.offset

    require(
      !dst0.isTranspose &&
      !src1.isTranspose &&
      rows0 == src1.rows && cols0 == src1.cols &&
      rows0 == src2.rows && cols0 == src2.cols
    )

    // Process all pairs.
    var nextGap0 = off0 + rows0
    var r        = 0
    var c        = 0
    while (iter2.hasNext) {
      val next = iter2.next()
      while (c < next._1._2) {
        while (off0 < nextGap0) {
          data0(off0) = fn1(data0(off0), data1(off1))
          off1 += 1
          off0 += 1
        }
        off1     += gap1
        off0     += gap0
        nextGap0  = off0 + rows0
        r         = 0
        c        += 1
      }
      while (r < next._1._1) {
        data0(off0) = fn1(data0(off0), data1(off1))
        off1 += 1
        off0 += 1
        r    += 1
      }
      data0(off0) = fn0(data0(off0), data1(off1), next._2)
      off1 += 1
      off0 += 1
      r    += 1
    }

    // If values remaining process them.
    val end0 = dst0.offset + cols0 * stride0
    while (off0 < end0) {
      while (off0 < nextGap0) {
        data0(off0) = fn1(data0(off0), data1(off1))
        off1 += 1
        off0 += 1
      }
      off1 += gap1
      off0 += gap0
      nextGap0 = off0 + rows0
    }
  }

  @inline
  final def transformPairs[T](dst0: DenseMatrix[T])
                             (fn: (Int, Int, T) => T)
  : Unit = {
    val data0   = dst0.data
    val stride0 = dst0.majorStride
    var off0    = dst0.offset
    val rows0   = dst0.rows
    val cols0   = dst0.cols

    if (dst0.isTranspose) {
      var r = 0
      while (r < rows0) {
        var c = 0
        while (c < cols0) {
          val tmp = off0 + c
          data0(tmp) = fn(r, c, data0(tmp))
          c += 1
        }
        off0 += stride0
        r    += 1
      }
    }
    else {
      var c = 0
      while (c < cols0) {
        var r = 0
        while (r < rows0) {
          val tmp = off0 + r
          data0(tmp) = fn(r, c, data0(tmp))
          r += 1
        }
        off0 += stride0
        c    += 1
      }
    }
  }

  @inline
  final def transformPairs[T, U](dst0: DenseMatrix[T],
                                 src1: DenseMatrix[U])
                                (fn: (Int, Int, T, U) => T)
  : Unit = {
    val data1   = src1.data
    val stride1 = src1.majorStride
    var off1    = src1.offset
    val data0   = dst0.data
    val stride0 = dst0.majorStride
    var off0    = dst0.offset
    val rows0   = dst0.rows
    val cols0   = dst0.cols

    require(
      !dst0.isTranspose &&
      !src1.isTranspose &&
      rows0 == src1.rows &&
      cols0 == src1.cols
    )

    var c = 0
    while (c < cols0) {
      var r = 0
      while (r < rows0) {
        val tmp0 = off0 + r
        val tmp1 = off1 + r
        data0(tmp0) = fn(r, c, data0(tmp0), data1(tmp1))
        r += 1
      }
      off1 += stride1
      off0 += stride0
      c    += 1
    }
  }

  @inline
  final def transformPairs[T, U, V](dst0: DenseMatrix[T],
                                    src1: DenseMatrix[U],
                                    src2: DenseMatrix[V])
                                   (fn: (Int, Int, T, U, V) => T)
  : Unit = {
    val data2   = src2.data
    val stride2 = src2.majorStride
    var offset2 = src2.offset
    val data1   = src1.data
    val stride1 = src1.majorStride
    var offset1 = src1.offset
    val data0   = dst0.data
    val stride0 = dst0.majorStride
    var offset0 = dst0.offset
    val rows0   = dst0.rows
    val cols0   = dst0.cols

    require(
      !dst0.isTranspose &&
      !src1.isTranspose &&
      !src2.isTranspose &&
      rows0 == src1.rows &&
      cols0 == src1.cols &&
      rows0 == src2.rows &&
      cols0 == src2.cols
    )

    var c = 0
    while (c < cols0) {
      var r = 0
      while (r < rows0) {
        val tmp0 = offset0 + r
        val tmp1 = offset1 + r
        val tmp2 = offset2 + r
        data0(tmp0) = fn(r, c, data0(tmp0), data1(tmp1), data2(tmp2))
        r += 1
      }
      offset2 += stride2
      offset1 += stride1
      offset0 += stride0
      c       += 1
    }
  }

  @inline
  final def transformPairs[T, U, V, W](dst0: DenseMatrix[T],
                                       src1: DenseMatrix[U],
                                       src2: DenseMatrix[V],
                                       src3: DenseMatrix[W])
                                      (fn: (Int, Int, T, U, V, W) => T)
  : Unit = {
    val data3   = src3.data
    val stride3 = src3.majorStride
    var offset3 = src3.offset
    val data2   = src2.data
    val stride2 = src2.majorStride
    var offset2 = src2.offset
    val data1   = src1.data
    val stride1 = src1.majorStride
    var offset1 = src1.offset
    val data0   = dst0.data
    val stride0 = dst0.majorStride
    var offset0 = dst0.offset
    val rows0   = dst0.rows
    val cols0   = dst0.cols

    require(
      !dst0.isTranspose &&
      !src1.isTranspose &&
      !src2.isTranspose &&
      rows0 == src1.rows &&
      cols0 == src1.cols &&
      rows0 == src2.rows &&
      cols0 == src2.cols &&
      rows0 == src3.rows &&
      cols0 == src3.cols
    )

    var c = 0
    while (c < cols0) {
      var r = 0
      while (r < rows0) {
        val tmp0 = offset0 + r
        val tmp1 = offset1 + r
        val tmp2 = offset2 + r
        val tmp3 = offset3 + r
        data0(tmp0) = fn(
          r, c, data0(tmp0), data1(tmp1), data2(tmp2), data3(tmp3)
        )
        r += 1
      }
      offset3 += stride3
      offset2 += stride2
      offset1 += stride1
      offset0 += stride0
      c       += 1
    }
  }

  @inline
  final def zip[T, U, V](src0: DenseMatrix[T],
                         src1: DenseMatrix[U])
                        (fn: (T, U) => V)
                        (implicit tagV: ClassTag[V])
  : Array[V] = {
    require(
      src0.isTranspose == src1.isTranspose &&
      src0.rows        == src1.rows        &&
      src0.cols        == src1.cols
    )

    val data1   = src1.data
    val gap1    = src1.majorStride - minor(src1)
    var offset1 = src1.offset
    val data0   = src0.data
    val min0    = minor(src0)
    val gap0    = src0.majorStride - min0
    var offset0 = src0.offset

    val result = new Array[V](src0.size)
    var i      = 0
    while (i < result.length) {
      val nextGap = i + min0
      while (i < nextGap) {
        result(i) = fn(data0(offset0), data1(offset1))
        offset1 += 1
        offset0 += 1
        i       += 1
      }
      offset1 += gap1
      offset0 += gap0
    }
    result
  }

  @inline
  final def zip[T, U, V](src0: CSCMatrix[T],
                         src1: CSCMatrix[U])
                        (fn: (T, U) => V)
                        (implicit tagV: ClassTag[V], zeroV: Zero[V])
  : CSCMatrix[V] = {
    // Shorthands for frequently sued variables.
    val offsets1 = src1.colPtrs
    val indices1 = src1.rowIndices
    val data1    = src1.data
    val offsets0 = src0.colPtrs
    val indices0 = src0.rowIndices
    val data0    = src0.data
    val rows0    = src0.rows
    val cols0    = src0.cols

    require(
      rows0 == src1.rows &&
      cols0 == src1.cols
    )

    // Allocate memory for results.
    val offsetsR = new Array[Int](offsets0.length)
    val usedR    = Math.min(src0.activeSize, src1.activeSize)
    val indicesR = new Array[Int](usedR)
    val dataR    = new Array[V](usedR)
    var offR     = 0

    // Process all pairs.
    var off0 = offsets0(0)
    var off1 = offsets1(0)
    var c    = 1
    while (c < offsets0.length) {
      val end0 = offsets0(c)
      val end1 = offsets1(c)

      while (off0 < end0 && off1 < end1) {
        val index0 = indices0(off0)
        val index1 = indices1(off1)

        if (index0 < index1) {
          off0 += 1
        }
        else if (index0 > index1) {
          off1 += 1
        }
        else {
          indicesR(offR) = index0
          dataR(offR)    = fn(data0(off0), data1(off1))
          offR += 1
          off0 += 1
          off1 += 1
        }
      }

      offsetsR(c) = offR
      off0        = end0
      off1        = end1
      c += 1
    }

    new CSCMatrix(dataR, rows0, cols0, offsetsR, indicesR)
  }

  @inline
  final def zipColumnVectors[T, U, V](src0: DenseMatrix[T],
                                      src1: DenseMatrix[U])
                                     (fn: (DenseVector[T], DenseVector[U]) => V)
                                     (implicit tagV: ClassTag[V])
  : Array[V] = {
    val result = new Array[V](src0.cols)
    foreachColumnVectorPair(
      src0,
      src1
    )((i, v0, v1) => result(i) = fn(v0, v1))
    result
  }

  @inline
  final def zipColumnVectors[T, U, V, W](src0: DenseMatrix[T],
                                         src1: DenseMatrix[U],
                                         src2: DenseMatrix[V])
                                        (fn: (DenseVector[T], DenseVector[U], DenseVector[V]) => W)
                                        (implicit tagW: ClassTag[W])
  : Array[W] = {
    val result = new Array[W](src0.cols)
    foreachColumnVectorPair(
      src0,
      src1,
      src2
    )((i, v0, v1, v2) => result(i) = fn(v0, v1, v2))
    result
  }

  @inline
  final def zipColumnVectors[T, U, V, W, X](src0: DenseMatrix[T],
                                            src1: DenseMatrix[U],
                                            src2: DenseMatrix[V],
                                            src3: DenseMatrix[W])
                                           (fn: (DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W]) => X)
                                           (implicit tagX: ClassTag[X])
  : Array[X] = {
    val result = new Array[X](src0.cols)
    foreachColumnVectorPair(
      src0,
      src1,
      src2,
      src3
    )((i, v0, v1, v2, v3) => result(i) = fn(v0, v1, v2, v3))
    result
  }

  @inline
  final def zipColumnVectorPairs[T, U, V](src0: DenseMatrix[T],
                                          src1: DenseMatrix[U])
                                         (fn: (Int, DenseVector[T], DenseVector[U]) => V)
                                         (implicit tagV: ClassTag[V])
  : Array[V] = {
    val result = new Array[V](src0.rows)
    foreachColumnVectorPair(
      src0,
      src1
    )((i, v0, v1) => result(i) = fn(i, v0, v1))
    result
  }

  @inline
  final def zipColumnVectorPairs[T, U, V, W](src0: DenseMatrix[T],
                                             src1: DenseMatrix[U],
                                             src2: DenseMatrix[V])
                                            (fn: (Int, DenseVector[T], DenseVector[U], DenseVector[V]) => W)
                                            (implicit tagW: ClassTag[W])
  : Array[W] = {
    val result = new Array[W](src0.rows)
    foreachColumnVectorPair(
      src0,
      src1,
      src2
    )((i, v0, v1, v2) => result(i) = fn(i, v0, v1, v2))
    result
  }

  @inline
  final def zipColumnVectorPairs[T, U, V, W, X](src0: DenseMatrix[T],
                                                src1: DenseMatrix[U],
                                                src2: DenseMatrix[V],
                                                src3: DenseMatrix[W])
                                               (fn: (Int, DenseVector[T], DenseVector[U], DenseVector[V], DenseVector[W]) => X)
                                               (implicit tagX: ClassTag[X])
  : Array[X] = {
    val result = new Array[X](src0.rows)
    foreachColumnVectorPair(
      src0,
      src1,
      src2,
      src3
    )((i, v0, v1, v2, v3) => result(i) = fn(i, v0, v1, v2, v3))
    result
  }

  @inline
  final def zipEx[T, U, V](src0: CSCMatrix[T],
                           src1: CSCMatrix[U])
                          (fn0: (T, U) => V, fn1: T => V, fn2: U => V)
                          (implicit tagV: ClassTag[V], zeroV: Zero[V])
  : CSCMatrix[V] = {
    // Shorthands for frequently sued variables.
    val offsets1 = src1.colPtrs
    val indices1 = src1.rowIndices
    val data1    = src1.data
    val offsets0 = src0.colPtrs
    val indices0 = src0.rowIndices
    val data0    = src0.data
    val rows0    = src0.rows
    val cols0    = src0.cols

    require(
      rows0 == src1.rows &&
      cols0 == src1.cols
    )

    // Allocate memory for results.
    val offsetsR = new Array[Int](offsets0.length)
    val usedR    = src0.activeSize + src1.activeSize
    val indicesR = new Array[Int](usedR)
    val dataR    = new Array[V](usedR)
    var offR     = 0

    // Process all pairs.
    var off0 = offsets0(0)
    var off1 = offsets1(0)
    var c    = 1
    while (c < offsets0.length) {
      val end0 = offsets0(c)
      val end1 = offsets1(c)

      while (off0 < end0 && off1 < end1) {
        val index0 = indices0(off0)
        val index1 = indices1(off1)

        if (index0 < index1) {
          indicesR(offR) = index0
          dataR(offR)    = fn1(data0(off0))
          offR += 1
          off0 += 1
        }
        else if (index0 > index1) {
          indicesR(offR) = index1
          dataR(offR)    = fn2(data1(off1))
          offR += 1
          off1 += 1
        }
        else {
          indicesR(offR) = index0
          dataR(offR)    = fn0(data0(off0), data1(off1))
          offR += 1
          off0 += 1
          off1 += 1
        }
      }

      while (off0 < end0) {
        indicesR(offR) = indices0(off0)
        dataR(offR)    = fn1(data0(off0))
        offR += 1
        off0 += 1
      }

      while (off1 < end1) {
        indicesR(offR) = indices1(off1)
        dataR(offR)    = fn2(data1(off1))
        offR += 1
        off1 += 1
      }

      offsetsR(c) = offR
      c += 1
    }

    if (offR != usedR) {
      new CSCMatrix(
        ArrayEx.take(dataR, offR),
        rows0,
        cols0,
        offsetsR,
        ArrayEx.take(indicesR, offR)
      )
    }
    else {
      new CSCMatrix(
        dataR,
        rows0,
        cols0,
        offsetsR,
        indicesR
      )
    }
  }

  @inline
  final def zipPairs[T, U, V](src0: DenseMatrix[T],
                              src1: DenseMatrix[U])
                             (fn: (Int, T, U) => V)
                             (implicit tagV: ClassTag[V])
  : Array[V] = {
    require(
      src0.isTranspose == src1.isTranspose &&
      src0.rows        == src1.rows        &&
      src0.cols        == src1.cols
    )

    val data1   = src1.data
    val gap1    = src1.majorStride - minor(src1)
    var offset1 = src1.offset
    val data0   = src0.data
    val min0    = minor(src0)
    val gap0    = src0.majorStride - min0
    var offset0 = src0.offset

    val result = new Array[V](src0.size)
    var i      = 0
    while (i < result.length) {
      val nextGap = i + min0
      while (i < nextGap) {
        result(i) = fn(i, data0(offset0), data1(offset1))
        offset1 += 1
        offset0 += 1
        i       += 1
      }
      offset1 += gap1
      offset0 += gap0
    }
    result
  }

  @inline
  final def zipRowVectors[T, U, V](src0: DenseMatrix[T],
                                   src1: DenseMatrix[U])
                                  (fn: (DenseVector[T], DenseVector[U]) => V)
                                  (implicit tagV: ClassTag[V])
  : Array[V] = {
    val result = new Array[V](src0.rows)
    foreachRowVectorPair(
      src0,
      src1
    )((i, v0, v1) => result(i) = fn(v0, v1))
    result
  }

  @inline
  final def fastZipRowPairs[T, U, V](src0: DenseMatrix[T],
                                     src1: DenseMatrix[U])
                                    (fn: (Int, DenseVector[T], DenseVector[U]) => V)
                                    (implicit tagV: ClassTag[V])
  : Array[V] = {
    val result = new Array[V](src0.rows)
    foreachRowVectorPair(
      src0,
      src1
    )((i, v0, v1) => result(i) = fn(i, v0, v1))
    result
  }

}
