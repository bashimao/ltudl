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

package object math {

  /**
    * "As" functions will try to use existing memory if possible.
    */
  /*
  final implicit class VectorFunctions[T](v: Vector[T]) {

    /*
    @inline
    def fastMapActive[U: ClassTag](fn: T => U)(implicit zero: Zero[U])
    : Vector[U] = v match {
      case v: DenseVector[T] =>
        v.fastMap(fn)
      case v: SparseVector[T] =>
        v.fastMapActive(fn)
      case _ =>
        throw new MatchError(v)
    }

    @inline
    def fastZip[U: ClassTag, V: ClassTag](other: Vector[U])(fn: (T, U) => V)
                                         (implicit zero0: Zero[T], zero1: Zero[U], zeroR: Zero[V])
    : Vector[V] = v match {
      case v: DenseVector[T] =>
        v.fastZip(other)(fn)
      case v: SparseVector[T] =>
        v.fastZip(other)(fn)
      case _ =>
        throw new MatchError(v)
    }

    @inline
    def fastZipEx[U: ClassTag, V: ClassTag](other: Vector[U])
                                           (fn0: (T, U) => V, fn1: T => V, fn2: U => V)
                                           (implicit zero: Zero[V])
    : Vector[V] = v match {
      case v: DenseVector[T] =>
        v.fastZipEx(other)(fn0, fn1)
      case v: SparseVector[T] =>
        v.fastZipEx(other)(fn0, fn1, fn2)
      case _ =>
        throw new MatchError(v)
    }

    @inline
    def fastZipPairs[U: ClassTag, V: ClassTag](other: Vector[U])(fn: (Int, T, U) => V)
                                              (implicit zero0: Zero[T], zero1: Zero[U], zeroR: Zero[V])
    : Vector[V] = v match {
      case v: DenseVector[T] =>
        v.fastZipPairs(other)(fn)
      case v: SparseVector[T] =>
        v.fastZipPairs(other)(fn)
      case _ =>
        throw new MatchError(v)
    }

    @inline
    def fastZipPairsEx[U: ClassTag, V: ClassTag](other: Vector[U])
                                                (fn0: (Int, T, U) => V, fn1: (Int, T) => V, fn2: (Int, U) => V)
                                                (implicit zero: Zero[V])
    : Vector[V] = v match {
      case v: DenseVector[T] =>
        v.fastZipPairsEx(other)(fn0, fn1)
      case v: SparseVector[T] =>
        v.fastZipPairsEx(other)(fn0, fn1, fn2)
      case _ =>
        throw new MatchError(v)
    }

    @inline
    def memoryUtilization(): Real = v match {
      case v: DenseVector[T] =>
        Real.one
      case v: SparseVector[T] =>
        v.memoryUtilization()
      case _ =>
        throw new MatchError(v)
    }
    */

    /*
    @inline
    def ++(other: Vector[T])(implicit classTag: ClassTag[T], zero: Zero[T])
    : Vector[T] = v match {
      case v: DenseVector[T] =>
        v ++ other
      case v: SparseVector[T] =>
        v ++ other
      case _ =>
        throw new MatchError(v)
    }

    @inline
    def :++(other: Vector[T])(implicit classTag: ClassTag[T], zero: Zero[T])
    : Vector[T] = v match {
      case v: DenseVector[T] =>
        v :++ other
      case v: SparseVector[T] =>
        v :++ other
      case _ =>
        throw new MatchError(v)
    }
    */

  }
  */


  /**
    * Breeze DenseVector type.
    */
  final implicit class DenseVectorFunctions[T](dv: DenseVector[T]) {

    /*
    @inline
    def toSparse(implicit classTag: ClassTag[T], zero: Zero[T])
    : SparseVector[T] = {
      val used1    = dv.length
      val indices1 = Array.ofDim[Int](used1)
      val data0    = dv.data
      val stride0  = dv.stride

      if (stride0 >= 0 && used1 == data0.length) {
        cforRange(0 until used1)(
          i => indices1(i) = i
        )
        new SparseVector(indices1, data0, used1)
      }
      else {
        // TODO: Could avoid some java range checks.
        val data1   = Array.ofDim[T](used1)
        val stride0 = dv.stride
        var offset0 = dv.offset
        var offset1 = 0
        while (offset1 < used1) {
          data1(offset1) = data0(offset0)
          offset0 += stride0
          indices1(offset1) = offset1
          offset1 += 1
        }
        new SparseVector(indices1, data1, used1)
      }
    }
    */

    /*
    @inline
    def transformEx[U, V](other: DenseMatrix[U], other2: DenseMatrix[V])
                         (fn: (T, U, V) => T)
    : Unit = {
      val data2   = other2.data
      val gap2    = other2.majorStride - other2.rows
      var offset2 = other2.offset
      val data1   = other.data
      val rows1   = other.rows
      val cols1   = other.cols
      val stride1 = other.majorStride
      val gap1    = stride1 - rows1
      var offset1 = other.offset
      val end1    = other.offset * stride1 * cols1
      val data0   = dv.data
      val stride0 = dv.stride
      val length0 = dv.length
      val end0    = dv.offset + stride0 * length0

      require(
        !other.isTranspose && !other2.isTranspose &&
        length0 == rows1 && length0 == other2.rows &&
        cols1 == other2.cols
      )

      while (offset1 < end1) {
        var offset0 = dv.offset
        while (offset0 != end0) {
          data0(offset0) = fn(data0(offset0), data1(offset1), data2(offset2))
          offset2 += 1
          offset1 += 1
          offset0 += stride0
        }
        offset2 += gap2
        offset1 += gap1
      }
    }

    @inline
    def transformRowsEx[U](other: DenseMatrix[U])(fn: (T, DenseVector[U]) => T)
    : Unit = {
      val data1   = other.data
      val stride1 = other.majorStride
      val cols1   = other.cols
      var offset1 = other.offset
      val data0   = dv.data
      val stride0 = dv.stride
      val length0 = dv.length
      var offset0 = dv.offset
      val end0    = dv.offset + stride0 * length0

      require(!other.isTranspose && length0 == other.rows)

      while (offset0 != end0) {
        fn(data0(offset0), new DenseVector(data1, offset1, stride1, cols1))
        offset1 += 1
        offset0 += stride0
      }
    }
    */

    /**
      * Normal vertical concatenation.
      */

    /*
    @inline
    def ++(other: Vector[T])(implicit classTag: ClassTag[T])
    : Array[T] = VectorEx.concat(dv, other)

    /**
      * Normal vertical concatenation.
      */
    @inline
    def ++(other: DenseVector[T])(implicit classTag: ClassTag[T])
    : Array[T] = VectorEx.concat(dv, other)

    /**
      * Normal vertical concatenation.
      */
    @inline
    def ++(other: SparseVector[T])
          (implicit classTag: ClassTag[T])
    : Array[T] = VectorEx.concat(dv, other)

    /**
      * Interleaving vertical concatenation.
      */
    @inline
    def :++(other: Vector[T])
           (implicit classTag: ClassTag[T])
    : Array[T] = VectorEx.interleave(dv, other)

    /**
      * Interleaving vertical concatenation.
      */
    @inline
    def :++(other: DenseVector[T])
           (implicit classTag: ClassTag[T])
    : Array[T] = VectorEx.interleave(dv, other)

    /**
      * Interleaving vertical concatenation.
      */
    @inline
    def :++(other: SparseVector[T])
           (implicit classTag: ClassTag[T])
    : Array[T] = VectorEx.interleave(dv, other)
    */

  }

  /*
  final implicit class TransposeDenseVectorFunctions[T](tdv: Transpose[DenseVector[T]]) {

    /*
    /**
      * Equivalent of Matlabs repmat function.
      *
      * @param times Number of copies of the row vector.
      * @return
      */
    def repeatV(times: Int): DMat = {
      // TODO: See other repeat function to see how to make this faster.
      val result = DMat.zeros(times, tdv.inner.length)
      result(*, ::).t := tdv.inner
      result
    }
    */

  }
  */

  final implicit class SparseVectorFunctions[T](sv: SparseVector[T]) {

    /*
    @inline
    def activeTopK[U](k: Int, scoreFn: ((Int, T)) => U)
                     (implicit ord: Ordering[U])
    : Iterable[(Int, T)] = TopK.apply[(Int, T), U](
      k, sv.activeIterator, scoreFn
    )
    */

    /*
    @inline
    def zip[U, V: ClassTag](other: Vector[U])(fn: (T, U) => V)
                           (implicit zeroT: Zero[T], zeroU: Zero[U], zeroV: Zero[V])
    : Vector[V] = other match {
      case other: DenseVector[U] =>
        zip(other)(fn)
      case other: SparseVector[U] =>
        zip(other)(fn)
      case _ =>
        throw new MatchError(other)
    }
    */

    /*
    @inline
    def fastZipEx[U: ClassTag, V: ClassTag](other: Vector[U])
                                           (fn0: (T, U) => V, fn1: T => V, fn2: U => V)
                                           (implicit zero: Zero[V])
    : Vector[V] = other match {
      case other: DenseVector[U] =>
        fastZipEx(other)(fn0, fn2)
      case other: SparseVector[U] =>
        fastZipEx(other)(fn0, fn1, fn2)
      case _ =>
        throw new MatchError(other)
    }
    */

    /*
    @inline
    def fastZipPairs[U: ClassTag, V: ClassTag](other: Vector[U])(fn: (Int, T, U) => V)
                                              (implicit zero0: Zero[T], zero1: Zero[U], zeroR: Zero[V])
    : Vector[V] = other match {
      case other: DenseVector[U] =>
        fastZipPairs(other)(fn)
      case other: SparseVector[U] =>
        fastZipPairs(other)(fn)
      case _ =>
        throw new MatchError(other)
    }
    */

    /*
    @inline
    def fastZipPairsEx[U: ClassTag, V: ClassTag](other: Vector[U])
                                                (fn0: (Int, T, U) => V, fn1: (Int, T) => V, fn2: (Int, U) => V)
                                                (implicit zero: Zero[V])
    : Vector[V] = other match {
      case other: DenseVector[U] =>
        fastZipPairsEx(other)(fn0, fn2)
      case other: SparseVector[U] =>
        fastZipPairsEx(other)(fn0, fn1, fn2)
      case _ =>
        throw new MatchError(other)
    }
    */

    /**
      * Normal vertical concatenation.
      */
    /*
    @inline
    def ++(other: DenseVector[T])
          (implicit tagT: ClassTag[T])
    : Array[T] = VectorEx.concat(sv, other)

    @inline
    def ++(other: SparseVector[T])
          (implicit tagT: ClassTag[T], zeroT: Zero[T])
    : SparseArray[T] = VectorEx.concatV(sv, other)

    /**
      * Normal vertical concatenation.
      */
    @inline
    def :++(other: DenseVector[T])
           (implicit classTag: ClassTag[T])
    : Array[T] = VectorEx.interleave(sv, other)

    @inline
    def :++(other: SparseVector[T])
           (implicit classTag: ClassTag[T], zero: Zero[T])
    : SparseArray[T] = VectorEx.interleave(sv, other)
    */
  }

  /*
  /**
    * "As" functions will try to use existing memory if possible.
    */
  final implicit class MatrixFunctions[T](m: Matrix[T]) {

    /*
    @inline
    def ++(other: Matrix[T])(implicit classTag: ClassTag[T], zero: Zero[T])
    : Matrix[T] = m match {
      case m: DenseMatrix[T] =>
        m ++ other
      case m: CSCMatrix[T] =>
        m ++ other
      case _ =>
        throw new MatchError(m)
    }

    @inline
    def :++(other: Matrix[T])(implicit classTag: ClassTag[T], zero: Zero[T])
    : Matrix[T] = m match {
      case m: DenseMatrix[T] =>
        m :++ other
      case m: CSCMatrix[T] =>
        m :++ other
      case _ =>
        throw new MatchError(m)
    }
    */

  }
  */

  /*
  final implicit class MatFunctions(m: Mat) {

    // ---------------------------------------------------------------------------
    //    Advanced statistics. (Very useful in debugger!)
    // ---------------------------------------------------------------------------
    def noValuesNonZero: Int = {
      var i = 0
      val iter = m.activeValuesIterator
      while (iter.hasNext) {
        if (iter.next() != Real.zero) {
          i += 1
        }
      }
      i
    }

    def noValuesValid: Int = m.size - noValuesNotANumber - noValuesInfinite

    def noValuesNotANumber: Int = {
      var i = 0
      val iter = m.activeValuesIterator
      while (iter.hasNext) {
        if (iter.next().isNaN) {
          i += 1
        }
      }
      i
    }

    def noValuesInfinite: Int = {
      var i = 0
      val iter = m.activeValuesIterator
      while (iter.hasNext) {
        if (iter.next().isInfinite) {
          i += 1
        }
      }
      i
    }

    def maxValue: Real = m match {
      case m: DMat => max(m)
      case m: SMat => max(m)
    }

    def maxValuePosition: (Int, Int) = argmax(m.asOrToDense)

    def minValue: Real = m match {
      case m: DMat => min(m)
      case m: SMat => min(m)
    }

    def minValuePosition: (Int, Int) = argmin(m.asOrToDense)

  }
  */

  /**
    * Breeze DenseVector type.
    */
  final implicit class DenseMatrixFunctions[T](dm: DenseMatrix[T]) {

    /*
    @inline
    def concatH(others: Traversable[DenseMatrix[T]])
               (implicit classTag: ClassTag[T], zero: Zero[T])
    : DenseMatrix[T] = {
      require(!dm.isTranspose)
      val cols = others.foldLeft(dm.cols)((res, m) => {
        require(!m.isTranspose)
        res + m.cols
      })

      val result = DenseMatrix.zeros[T](dm.rows, cols)
      var c1     = dm.cols
      result(::, 0 until c1) := dm
      others.foreach(m => {
        val c0 = c1
        c1 += m.cols
        result(::, c0 until c1) := m
      })
      result
    }
    */

    /*
    @inline
    def concatH(others: Array[DenseMatrix[T]])
               (implicit classTag: ClassTag[T], zero: Zero[T])
    : DenseMatrix[T] = {
      require(!dm.isTranspose)
      val cols = others.fastFoldLeft(dm.cols)((res, m) => {
        require(!m.isTranspose)
        res + m.cols
      })

      val result = DenseMatrix.zeros[T](dm.rows, cols)
      var c1     = dm.cols
      result(::, 0 until c1) := dm
      others.fastForeach(m => {
        val c0 = c1
        c1 += m.cols
        result(::, c0 until c1) := m
      })
      result
    }
    */

    /*

    @inline
    def fastForeachRowEx[U](other: DenseVector[U])
                           (fn: (DenseVector[T], U) => Unit)
    : Unit = {
      val data1   = other.data
      val stride1 = other.stride
      var offset1 = other.offset
      val data0   = dm.data
      val stride0 = dm.majorStride
      var offset0 = dm.offset
      val rows0   = dm.rows
      val cols0   = dm.cols
      val end0    = dm.offset + rows0

      require(!dm.isTranspose && rows0 == other.length)

      while (offset0 < end0) {
        fn(new DenseVector(data0, offset0, stride0, cols0), data1(offset1))
        offset1 += stride1
        offset0 += 1
      }
    }
    */

    /*

    @inline
    def fastZipCols[U, V: ClassTag](other: DenseVector[U])(fn: (T, U) => V)
    : Array[V] = {
      val stride1 = other.stride
      val data1   = other.data
      val data0   = dm.data
      val rows0   = dm.rows
      val gap0    = dm.majorStride - rows0

      require(!dm.isTranspose && rows0 == other.length)

      val result  = Array.ofDim[V](dm.size)
      var offset0 = dm.offset
      var i       = 0
      while (i < result.length) {
        var offset1 = other.offset
        val nextGap = i + rows0
        while (i < nextGap) {
          result(i) = fn(data0(offset0), data1(offset1))
          offset1 += stride1
          offset0 += 1
          i       += 1
        }
        offset0 += gap0
      }
      result
    }

    @inline
    def fastZipCols[U, V, W: ClassTag](other:  DenseVector[U],
                                       other2: DenseVector[V])
                                      (fn: (T, U, V) => W)
    : Array[W] = {
      val data2   = other2.data
      val stride2 = other2.stride
      val data1   = other.data
      val stride1 = other.stride
      val data0   = dm.data
      val rows0   = dm.rows
      val gap0    = dm.majorStride - rows0

      require(
        !dm.isTranspose && rows0 == other.length && rows0 == other2.length
      )

      val result  = Array.ofDim[W](dm.size)
      var offset0 = dm.offset
      var i       = 0
      while (i < result.length) {
        var offset2 = other2.offset
        var offset1 = other.offset
        val nextGap = i + rows0
        while (i < nextGap) {
          result(i) = fn(data0(offset0), data1(offset1), data2(offset2))
          offset2 += stride2
          offset1 += stride1
          offset0 += 1
          i       += 1
        }
        offset0 += gap0
      }
      result
    }
    */

    /*

    @inline
    def subMatrixOffset(row0: Int, col0: Int): Int = {
      if (dm.isTranspose) {
        dm.offset + row0 * dm.majorStride + col0
      }
      else {
        dm.offset + col0 * dm.majorStride + row0
      }
    }
    */

    /*
    @inline
    def transformEx[U, V](other: DenseMatrix[U], other2: DenseVector[V])
                         (fn: (T, U, V) => T)
    : Unit = {
      val data2   = other2.data
      val stride2 = other2.stride
      val data1   = other.data
      val gap1    = other.majorStride - other.rows
      var offset1 = other.offset
      val data0   = dm.data
      val rows0   = dm.rows
      val cols0   = dm.cols
      val stride0 = dm.majorStride
      val gap0    = stride0 - rows0
      var offset0 = dm.offset
      val end0    = dm.offset + stride0 * cols0

      require(
        !dm.isTranspose && !other.isTranspose &&
        rows0 == other.rows && rows0 == other2.length &&
        cols0 == other.cols
      )

      while (offset0 < end0) {
        var offset2 = other2.offset
        val nextGap = offset0 + rows0
        while (offset0 < nextGap) {
          data0(offset0) = fn(data0(offset0), data1(offset1), data2(offset2))
          offset2 += stride2
          offset1 += 1
          offset0 += 1
        }
        offset1 += gap1
        offset0 += gap0
      }
    }

    @inline
    def transformEx[U, V, W](other:  DenseMatrix[U],
                             other2: DenseMatrix[V],
                             other3: DenseVector[W])
                            (fn: (T, U, V, W) => T)
    : Unit = {
      val data3   = other3.data
      val stride3 = other3.stride
      val data2   = other2.data
      val gap2    = other2.majorStride - other2.rows
      var offset2 = other2.offset
      val data1   = other.data
      val gap1    = other.majorStride  - other.rows
      var offset1 = other.offset
      val data0   = dm.data
      val rows0   = dm.rows
      val cols0   = dm.cols
      val stride0 = dm.majorStride
      val gap0    = stride0 - rows0
      var offset0 = dm.offset
      val end0    = dm.offset + stride0 * cols0

      require(
        !dm.isTranspose && !other.isTranspose && !other2.isTranspose &&
        rows0 == other.rows && rows0 == other2.rows && rows0 == other3.length &&
        cols0 == other.cols && cols0 == other2.cols
      )

      while (offset0 < end0) {
        var offset3 = other3.offset
        val nextGap = offset0 + rows0
        while (offset0 < nextGap) {
          data0(offset0) = fn(
            data0(offset0),
            data1(offset1),
            data2(offset2),
            data3(offset3)
          )
          offset3 += stride3
          offset2 += 1
          offset1 += 1
          offset0 += 1
        }
        offset2 += gap2
        offset1 += gap1
        offset0 += gap0
      }
    }
    */

    /*
    /**
      * Vertical concatenation. (columns)
      */
    @inline
    def ++(other: Matrix[T])(implicit classTag: ClassTag[T], zero: Zero[T])
    : Matrix[T] = other match {
      case other: DenseMatrix[T] =>
        dm ++ other
      case other: CSCMatrix[T] =>
        dm ++ other
      case _ =>
        throw new MatchError(other)
    }
    */

    /*
    /**
      * Vertical concatenation (values, interleaving).
      */
    @inline
    def :++(other: Matrix[T])(implicit classTag: ClassTag[T], zero: Zero[T])
    : Matrix[T] = other match {
      case other: DenseMatrix[T] =>
        dm :++ other
      case other: CSCMatrix[T] =>
        dm :++ other
      case _ =>
        throw new MatchError(other)
    }
    */

    /*
    @inline
    def endOffset: Int = {
      val offset = dm.offset
      val stride = dm.majorStride
      val rows   = dm.rows
      val cols   = dm.cols
      if (rows == 0 || cols == 0) {
        offset
      }
      else if (dm.isTranspose) {
        offset + (rows - 1) * stride + cols
      }
      else {
        offset + (cols - 1) * stride + rows
      }
    }
    */

    /*
    @inline
    def demultiplexCols(sequenceLength: Int)
                       (implicit classTag: ClassTag[T], zero: Zero[T])
    : DenseMatrix[T] = {
      val result = DenseMatrix.zeros[T](dm.rows, dm.cols)
      demultiplexCols(sequenceLength, result)
      result
    }

    // TODO: Add optimized version to Array and use it!
    @inline
    def demultiplexCols(sequenceLength: Int, result: DenseMatrix[T])
    : Unit = fastForeachColEx(result)(
      _.demultiplex(sequenceLength, _)
    )
    */

    /*
    def fastForeachPairEx[U: ClassTag](other: Matrix[U])(fn: (T, U) => Unit)
    : Unit = other match {
      case other: DenseMatrix[U] =>
        fastForeachEx(other)(fn)
      case other: CSCMatrix[U]  =>
        fastForeachPairEx(other)(fn)
      case _ =>
        throw new IllegalArgumentException
    }

    def fastForeachPairEx[U](other: CSCMatrix[U])(fn: (T, U) => Unit): Unit = {
      // TODO: Why not use the underlying data structure directly?
      debug_req(dm.rows == other.rows)
      debug_req(dm.cols == other.cols)
      val iter = other.activeIterator
      while (iter.hasNext) {
        val ((r, c), value) = iter.next()
        fn(dm.unsafeValueAt(r, c), value)
      }
    }
    */

    /*
    /**
     * We explicitly constraint the types T and U as hints. This allows the
     * JVM to transform virtual into explicit function calls.
     *
     * @param fn Function to execute on each item.
     * @param fn2 Will automatically be resolved from namespace fn.
     * @tparam V Parent UFunc object type.
     * @tparam W The sub-function to execute.
     */
    @inline
    def fastMapFn[V <: UFunc, W <: UFunc.UImpl[V, T, T]](fn: V)
                                                        (implicit fn2: W, lassTag: ClassTag[T], zero: Zero[T])
    : DenseMatrix[T] = dm.fastMap(fn2(_))

    /**
     * We explicitly constraint the types T and U as hints. This allows the
     * JVM to transform virtual into explicit function calls.
     *
     * @param fn Function to execute on each item.
     * @param fn2 Will automatically be resolved from namespace fn.
     * @tparam V Parent UFunc object type.
     * @tparam W The sub-function to execute.
     */
    @inline
    def fastMapFn[V <: UFunc, W <: UFunc.UImpl2[V, T, X, T], X](fn: V, v2: X)
                                                               (implicit fn2: W, lassTag: ClassTag[T], zero: Zero[T])
    : DenseMatrix[T] = dm.fastMap(fn2(_, v2))

    /**
     * We explicitly constraint the types T and U as hints. This allows the
     * JVM to transform virtual into explicit function calls.
     *
     * @param fn Function to execute on each item.
     * @param fn2 Will automatically be resolved from namespace fn.
     * @tparam V Parent UFunc object type.
     * @tparam W The sub-function to execute.
     */
    @inline
    def fastMapFn[V <: UFunc, W <: UFunc.UImpl3[V, T, X, Y, T], X, Y](fn: V, v2: X, v3: Y)
                                                                     (implicit fn2: W, lassTag: ClassTag[T], zero: Zero[T])
    : DenseMatrix[T] = dm.fastMap(fn2(_, v2, v3))

    /**
     * We explicitly constraint the types T and U as hints. This allows the
     * JVM to transform virtual into explicit function calls.
     *
     * @param fn Function to execute on each item.
     * @param fn2 Will automatically be resolved from namespace fn.
     * @tparam V Parent UFunc object type.
     * @tparam W The sub-function to execute.
     */
    @inline
    def fastMapFn[V <: UFunc, W <: UFunc.UImpl4[V, T, X, Y, Z, T], X, Y, Z](fn: V, v2: X, v3: Y, v4: Z)
                                                                           (implicit fn2: W, classTag: ClassTag[T], zero: Zero[T])
    : DenseMatrix[T] = dm.fastMap(fn2(_, v2, v3, v4))
    */

    /*
    @inline
    def fastMapColsParallel[U: ClassTag](fn: DenseVector[T] => U): Array[U] = {
      require(!dm.isTranspose)

      val rows   = dm.rows
      val stride = dm.majorStride
      val data   = dm.data
      var offset = dm.offset

      val result = new ParArray[DenseVector[T]](dm.cols)
      var i      = 0
      while (i < result.length) {
        result(i) = new DenseVector(data, offset, 1, rows)
        offset += stride
        i      += 1
      }

      result.map(fn).toArray
    }
    */

    /*
    @inline
    def fill(fn: (Int, Int) => T): Unit = {
      val rows   = dm.rows
      val cols   = dm.cols
      val stride = dm.majorStride
      val data   = dm.data
      var offset = dm.offset
      if (dm.isTranspose) {
        val gap = stride - cols

        cfor(0)(_ < rows, _ + 1)(j => {
          cfor(0)(_ < cols, _ + 1)(i => {
            data(offset) = fn(i, j)
            offset += 1
          })
          offset += gap
        })
      }
      else {
        val gap = stride - rows

        cfor(0)(_ < cols, _ + 1)(j => {
          cfor(0)(_ < rows, _ + 1)(i => {
            data(offset) = fn(i, j)
            offset += 1
          })
          offset += gap
        })
      }
    }*/

    /*
    @inline
    def gap: Int = dm.majorStride - minor
    */

    /*
    @inline
    def head: T = {
      assume(dm.size > 0)
      dm.data(dm.offset)
    }

    @inline
    def last: T = dm.data(lastOffset)

    @inline
    def lastOffset: Int = {
      assume(dm.size > 0)
      dm.linearIndex(dm.rows - 1, dm.cols - 1)
    }
    */

    /*
    @inline
    def multiplexCols(sequenceLength: Int)
                     (implicit classTag: ClassTag[T], zero: Zero[T])
    : DenseMatrix[T] = {
      val result = DenseMatrix.zeros[T](dm.rows, dm.cols)
      demultiplexCols(sequenceLength, result)
      result
    }

    // TODO: Add optimized version to Array and use it!
    @inline
    def multiplexCols(sequenceLength: Int, result: DenseMatrix[T])
    : Unit = fastForeachColEx(result)(
      _.demultiplex(sequenceLength, _)
    )
    */

    /*
    /**
     * A more advanced version of updateEach. What this basically does is a
     * zip.inPlace with each column.
     */
    def transformColsEx[U: ClassTag](other: DenseVector[U])(fn: (T, U) => T)
    : Unit = {
      debug_req(dm.rows == other.length)

      val data1 = other.data
      val data0 = dm.data
      val min0  = dm.minor
      val gap0  = dm.majorStride - min0
      val end0  = dm.endOffset

      var offset0 = dm.offset
      while (offset0 < end0) {
        var offset1 = other.offset
        val nextGap = offset0 + min0
        while (offset0 < nextGap) {
          data0(offset0) = fn(data0(offset0), data1(offset1))
          offset1 += other.stride
          offset0 += 1
        }
        offset0 += gap0
      }
    }

    /**
     * A more advanced version of updateEach. What this basically does is a
     * zip.inPlace with each column.
     */
    def transformRowsEx[U: ClassTag](other: DenseVector[U])(fn: (T, U) => T)
    : Unit = {
      debug_req(dm.rows == other.length)

      val data1 = other.data
      val data0 = dm.data
      val min0  = dm.minor
      val gap0  = dm.majorStride - min0
      val end0  = dm.endOffset

      var offset0 = dm.offset
      while (offset0 < end0) {
        var offset1 = other.offset
        val nextGap = offset0 + min0
        while (offset0 < nextGap) {

          data0(offset0) = fn(data0(offset0), data1(offset1))
          offset1 += other.stride
          offset0 += 1
        }
        offset0 += gap0
      }
    }*/

    /*
    // TODO: For some reason dm(::, *).reduce does not work. Don't ask me why.
    def reduceLeftCols[U](result: DenseVector[U])(fn: (U, T) => U): Unit = {
      debug_req(result.length == dm.cols)
      var offset = result.offset
      var i = 0
      while (i < dm.cols) {
        result.data(offset) = dm(::, i).red(fn)
        offset += result.stride
        i      += 1
      }
    }
    */

  }

  /*
  final implicit class DMatFunctions(dm: DMat) {

    /*
    /**
      * Performs sum(dm(*, ::)) but avoids allocation of temporary array.
      */
    @inline
    def addColsTo(result: DVec): Unit = {
      // TODO: Check if this is really faster.. Can't believe. Should not have so many bias weights that memory allocation matters...
      // TODO: Shouldn't the broadcasting operator also nice properties?

      // TODO: Could do this much nicer if breeze would implement CanForeachValues for BroadcastedColumns.
      var i = 0
      while (i < dm.cols) {
        result += dm(::, i)
        i      += 1
      }
    }

    /**
      * Performs sum(dm(::, *)) but avoids allocation of temporary array.
      */
    @inline
    def addRowsTo(result: DVec): Unit = {
      // TODO: Check if this is really faster.. Can't believe. Should not have so many bias weights that memory allocation matters...
      // TODO: Shouldn't the broadcasting operator also nice properties?

      // TODO: Could do this much nicer if breeze would implement CanForeachValues for BroadcastedColumns.
      var i = 0
      while (i < dm.rows) {
        result += dm(i, ::).t
        i      += 1
      }
    }

    @inline
    def addToCols(other: DVec)
    : Unit = BLAS.axpyEx(Real.one, other, dm)
    */

  }
  */

  final implicit class SparseMatrixFunctions[T](sm: CSCMatrix[T]) {

    /*
    @inline
    def activeTopK[U](k: Int, scoreFn: (((Int, Int), T)) => U)
                     (implicit ord: Ordering[U])
    : Iterable[((Int, Int), T)] = TopK.apply[((Int, Int), T), U](
      k, sm.activeIterator, scoreFn
    )
    */

    /*

    // TODO: Slow!
    @inline
    def concatH(others: Traversable[CSCMatrix[T]])
               (implicit classTag: ClassTag[T], zero: Zero[T])
    : CSCMatrix[T] = {
      val result = CSCMatrix.zeros[T](
        sm.rows, others.foldLeft(sm.cols)(_ ?+ _.cols)
      )
      sm.fastForeachActivePair(result.update)
      var c0 = sm.cols
      others.foreach(m => {
        m.fastForeachActivePair((r, c, v) => result.update(r, c0 + c, v))
        c0 += m.cols
      })
      result
    }

    // TODO: Slow!
    @inline
    def concatH(others: Array[CSCMatrix[T]])
               (implicit classTag: ClassTag[T], zero: Zero[T])
    : CSCMatrix[T] = {
      val result = CSCMatrix.zeros[T](
        sm.rows, others.fastFoldLeft(sm.cols)(_ ?+ _.cols)
      )
      sm.fastForeachActivePair(result.update)
      var c0 = sm.cols
      others.fastForeach(m => {
        m.fastForeachActivePair((r, c, v) => result.update(r, c0 + c, v))
        c0 += m.cols
      })
      result
    }
    */

    /*
    /**
      * Vertical concatenation. (columns)
      */
    def ++(other: DenseMatrix[T])
          (implicit classTag: ClassTag[T])
    : Array[T] = MatrixEx.concatH(sm, other)

    /**
      * Vertical concatenation. (columns)
      */
    def ++(other: CSCMatrix[T])
          (implicit classTag: ClassTag[T])
    : CSCMatrix[T] = MatrixEx.interleaveV(sm, other)

    /**
      * Vertical concatenation (values, interleaving).
      */
    def :++(other: DenseMatrix[T])
           (implicit classTag: ClassTag[T])
    : Array[T] = MatrixEx.interleaveV(sm, other)

    /**
      * Vertical concatenation (values, interleaving).
      */
    def :++(other: CSCMatrix[T])
           (implicit classTag: ClassTag[T], zero: Zero[T])
    : CSCMatrix[T] = MatrixEx.interleaveV(sm, other)
    */

  }

}
