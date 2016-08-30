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

import breeze.collection.mutable.SparseArray
import breeze.linalg.{CSCMatrix, DenseMatrix, DenseVector, SparseVector, Transpose, Vector}
import breeze.storage.Zero

import scala.reflect._

object VectorEx {

  @inline
  final def add(dst0: DenseVector[Real],
                src1: Real)

  : Unit = ArrayEx.add(
    dst0.data, dst0.offset, dst0.stride,
    src1,
    dst0.length
  )

  @inline
  final def add(dst0: DenseVector[Real],
                src1: DenseVector[Real])

  : Unit = {
    require(dst0.length == src1.length)
    ArrayEx.add(
      dst0.data, dst0.offset, dst0.stride,
      src1.data, src1.offset, src1.stride,
      dst0.length
    )
  }

  @inline
  final def add(alpha: Real,
                dst0:  DenseVector[Real],
                src1:  DenseVector[Real])
  : Unit = {
    require(dst0.length == src1.length)
    ArrayEx.add(
      alpha,
      dst0.data, dst0.offset, dst0.stride,
      src1.data, src1.offset, src1.stride,
      dst0.length
    )
  }

  @inline
  final def add(dst0: DenseVector[Real],
                beta: Real,
                src1: DenseVector[Real])
  : Unit = {
    require(dst0.length == src1.length)
    ArrayEx.add(
      dst0.data, dst0.offset, dst0.stride,
      beta,
      src1.data, src1.offset, src1.stride,
      dst0.length
    )
  }

  @inline
  final def add(alpha: Real,
                dst0:  DenseVector[Real],
                beta:  Real,
                src1:  DenseVector[Real])
  : Unit = {
    require(dst0.length == src1.length)
    ArrayEx.add(
      alpha,
      dst0.data, dst0.offset, dst0.stride,
      beta,
      src1.data, src1.offset, src1.stride,
      dst0.length
    )
  }

  /**
    * Similar to toArray. But avoids allocation.
    */
  @inline
  final def asArray[T](src0: DenseVector[T])
  : Array[T] = {
    require(src0.length == src0.data.length)
    src0.data
  }

  /**
    * Differs from "asDenseMatrix" which creates a 1 x N matrix for internal
    * memory alignment reasons. This one will create a N x 1 matrix instead,
    * which is what we usually need and since we usually work with stride 1
    * this should almost always work properly as well.
    */
  @inline
  final def asMatrix[T](src0: DenseVector[T])
  : DenseMatrix[T] = {
    require(src0.stride == 1)
    new DenseMatrix[T](
      src0.length,
      1,
      src0.data,
      src0.offset,
      src0.length
    )
  }

  @inline
  final def asMatrix[T](src0: DenseVector[T], rows: Int)
  : DenseMatrix[T] = asMatrix(src0, rows, src0.length / rows)

  // TODO: Do this without creating an extra object.
  @inline
  final def asMatrix[T](src0: DenseVector[T], rows: Int, cols: Int)
  : DenseMatrix[T] = {
    require(src0.stride == 1 && src0.length == rows * cols)
    new DenseMatrix[T](
      rows,
      cols,
      src0.data,
      src0.offset,
      rows
    )
  }

  @inline
  def asOrToArray[T](src0: Vector[T])
                    (implicit tagT: ClassTag[T])
  : Array[T] = src0 match {
    case vector: DenseVector[T] =>
      asOrToArray(vector)
    case vector: SparseVector[T] =>
      toArray(vector)
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def asOrToArray[T](src0: DenseVector[T])
                          (implicit tagT: ClassTag[T])
  : Array[T] = {
    if (src0.length == src0.data.length) {
      src0.data
    }
    else {
      src0.toArray
    }
  }

  @inline
  final def asOrToDenseVector[T](src0: Vector[T])
                                (implicit tagT: ClassTag[T])
  : DenseVector[T] = src0 match {
    case src0: DenseVector[T] =>
      src0
    case src0: SparseVector[T] =>
      toDense(src0)
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def concat[T](dst0: Array[T], offset0: Int, stride0: Int,
                      src1: DenseVector[T],
                      src2: Vector[T])
  : Unit = src2 match {
    case src2: DenseVector[T] =>
      concat(
        dst0, offset0, stride0,
        src1,
        src2
      )
    case src2: SparseVector[T] =>
      concat(
        dst0, offset0, stride0,
        src1,
        src2
      )
    case _ =>
      throw new MatchError(src2)
  }

  @inline
  final def concat[T](dst0: Array[T], offset0: Int, stride0: Int,
                      src1: DenseVector[T],
                      src2: DenseVector[T])
  : Unit = ArrayEx.concatEx(
    dst0,      offset0,     stride0,
    src1.data, src1.offset, src1.stride,
    src1.length,
    src2.data, src2.offset, src2.stride,
    src2.length
  )

  @inline
  final def concat[T](dst0: Array[T], offset0: Int, stride0: Int,
                      src1: DenseVector[T],
                      src2: SparseVector[T])
  : Unit = ArrayEx.concatEx(
    dst0,      offset0,     stride0,
    src1.data, src1.offset, src1.stride,
    src1.length,
    src2.array
  )

  @inline
  final def concat[T](dst0: Array[T], offset0: Int, stride0: Int,
                      src1: SparseVector[T],
                      src2: DenseVector[T])
  : Unit = ArrayEx.concatEx(
    dst0,      offset0,     stride0,
    src1.array,
    src2.data, src2.offset, src2.stride,
    src2.length
  )

  @inline
  final def concat[T](src0: DenseVector[T],
                      src1: DenseVector[T])
                     (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](src0.length + src1.length)
    concat(
      result, 0, 1,
      src0,
      src1
    )
    result
  }

  @inline
  final def concat[T](dst0: DenseVector[T],
                      src1: DenseVector[T],
                      src2: Vector[T])
  : Unit = src2 match {
    case src2: DenseVector[T] =>
      concat(
        dst0,
        src1,
        src2
      )
    case src2: SparseVector[T] =>
      concat(
        dst0,
        src1,
        src2
      )
    case _ =>
      throw new MatchError(src2)
  }

  @inline
  final def concat[T](dst0: DenseVector[T],
                      src1: DenseVector[T],
                      src2: DenseVector[T])
  : Unit = {
    require(dst0.length == src1.length + src2.length)
    concat(
      dst0.data, dst0.offset, dst0.stride,
      src1,
      src2
    )
  }

  @inline
  final def concat[T](dst0: DenseVector[T],
                      src1: DenseVector[T],
                      src2: SparseVector[T])
  : Unit = {
    require(dst0.length == src1.length + src2.length)
    concat(
      dst0.data, dst0.offset, dst0.stride,
      src1,
      src2
    )
  }

  @inline
  final def concat[T](dst0: DenseVector[T],
                      src1: SparseVector[T],
                      src2: DenseVector[T])
  : Unit = {
    require(dst0.length == src1.length + src2.length)
    concat(
      dst0.data, dst0.offset, dst0.stride,
      src1,
      src2
    )
  }

  @inline
  final def concat[T](src0: SparseVector[T],
                      src1: SparseVector[T])
                     (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : SparseArray[T] = ArrayEx.concat(
    src0.array,
    src1.array
  )

  @inline
  final def concatLateral[T](src0: SparseVector[T],
                             src1: SparseVector[T])
                            (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : CSCMatrix[T] = {
    require(src0.length == src1.length)
    val result = CSCMatrix.zeros[T](src0.length, 2)
    foreachActivePair(
      src0
    )(result.update(_, 0, _))
    foreachActivePair(
      src1
    )(result.update(_, 1, _))
    result
  }

  @inline
  final def copy[T](dst0: Array[T], offset0: Int, stride0: Int,
                    src1: DenseVector[T])
  : Unit = ArrayEx.set(
    dst0,      offset0,     stride0,
    src1.data, src1.offset, src1.stride,
    src1.length
  )

  @inline
  final def copy[T](dst0: DenseVector[T],
                    src1: Array[T], offset1: Int, stride1: Int)
  : Unit = ArrayEx.set(
    dst0.data, dst0.offset, dst0.stride,
    src1,      offset1,     stride1,
    dst0.length
  )

  @inline
  final def copy[T](dst0: DenseVector[T],
                    src1: DenseVector[T])
                   (implicit tagT: ClassTag[T])
  : Unit = {
    require(dst0.length == src1.length)
    ArrayEx.set(
      dst0.data, dst0.offset, dst0.stride,
      src1.data, src1.offset, src1.stride,
      dst0.length
    )
  }

  @inline
  final def count[T](src0: Vector[T])
                    (predicate: T => Boolean)
  : Int = src0 match {
    case src0: DenseVector[T] =>
      count(
        src0
      )(predicate)
    case src0: SparseVector[T] =>
      count(
        src0
      )(predicate)
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def count[T](src0: DenseVector[T])
                    (predicate: T => Boolean)
  : Int = ArrayEx.count(
    src0.data, src0.offset, src0.stride,
    src0.length
  )(predicate)

  @inline
  final def count[T](src0: SparseVector[T])
                    (predicate: T => Boolean)
  : Int = ArrayEx.count(
    src0.array
  )(predicate)

  @inline
  final def countActive[T](src0: Vector[T])
                          (predicate: T => Boolean)
  : Int = src0 match {
    case src0: DenseVector[T] =>
      countActive(
        src0
      )(predicate)
    case src0: SparseVector[T] =>
      countActive(
        src0
      )(predicate)
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def countActive[T](src0: DenseVector[T])
                          (predicate: T => Boolean)
  : Int = count(
    src0
  )(predicate)

  @inline
  final def countActive[T](src0: SparseVector[T])
                          (predicate: T => Boolean)
  : Int = ArrayEx.countActive(
    src0.array
  )(predicate)

  @inline
  final def countActiveApprox[T](src0:   Vector[T],
                                 rng:       PseudoRNG,
                                 noSamples: Int)
                                (predicate: T => Boolean)
  : Int = src0 match {
    case src0: DenseVector[T] =>
      countActiveApprox(
        src0,
        rng,
        noSamples
      )(predicate)
    case src0: SparseVector[T] =>
      countActiveApprox(
        src0,
        rng,
        noSamples
      )(predicate)
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def countActiveApprox[T](src0: DenseVector[T],
                                 rng:       PseudoRNG,
                                 noSamples: Int)
                                (predicate: T => Boolean)
  : Int = countApprox(
    src0,
    rng,
    noSamples
  )(predicate)

  @inline
  final def countActiveApprox[T](src0: SparseVector[T],
                                 rng:       PseudoRNG,
                                 noSamples: Int)
                                (predicate: T => Boolean)
  : Int = ArrayEx.countActiveApprox(
    src0.array,
    rng,
    noSamples
  )(predicate)

  @inline
  final def countApprox[T](src0: Vector[T],
                           rng:       PseudoRNG,
                           noSamples: Int)
                          (predicate: T => Boolean)
  : Int = src0 match {
    case src0: DenseVector[T] =>
      countApprox(
        src0,
        rng,
        noSamples
      )(predicate)
    case src0: SparseVector[T] =>
      countApprox(
        src0,
        rng,
        noSamples
      )(predicate)
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def countApprox[T](src0: SparseVector[T],
                           rng:       PseudoRNG,
                           noSamples: Int)
                          (predicate: T => Boolean)
  : Int = ArrayEx.countApprox(
    src0.array,
    rng,
    noSamples
  )(predicate)

  @inline
  final def countApprox[T](src0:DenseVector[T],
                           rng:       PseudoRNG,
                           noSamples: Int)
                          (predicate: T => Boolean)
  : Int = ArrayEx.countApprox(
    src0.data, src0.offset, src0.stride,
    src0.length,
    rng,
    noSamples
  )(predicate)

  @inline
  final def diffL2NormSq(src0: DenseVector[Real],
                         src1: Real)
  : Real = ArrayEx.diffL2NormSq(
    src0.data, src0.offset, src0.stride,
    src1,
    src0.length
  )

  /**
    * Syntax sugar for dot product.
    * (Actually, the result type of the generic implementation of Breeze is not
    * resolved properly into a Real by IntelliJ. That is because the type of
    * "That" is not determinable (why?!). So this even adds some value. YAY!)
    * Don't believe me? Replace the following line with:
    * (dv dot other) * Real.one
    * wait a second and the "*" will become red.
    */
  // TODO: Look at usages and think about implementing magnitude function.
  @inline
  final def dot(src0: DenseVector[Real],
                src1: DenseVector[Real])
  : Real = {
    require(src0.length == src1.length)
    ArrayEx.dot(
      src0.data, src0.offset, src0.stride,
      src1.data, src1.offset, src1.stride,
      src0.length
    )
  }

  @inline
  final def endOffset[T](src0: DenseVector[T])
  : Int = src0.offset + src0.length * src0.stride

  @inline
  final def exists[T](src0: DenseVector[T])
                     (predicate: T => Boolean)
  : Boolean = ArrayEx.exists(
    src0.data, src0.offset, src0.stride,
    src0.length
  )(predicate)

  @inline
  final def fill[T](length: Int, value: T)
                   (implicit tagT: ClassTag[T])
  : DenseVector[T] = DenseVector(ArrayEx.fill(length, value))

  @inline
  final def fill[T](length: Int)
                   (fn: => T)
                   (implicit tagT: ClassTag[T])
  : DenseVector[T] = DenseVector.fill(length)(fn)

  /**
    * Other method quite slow... Why are the Breeze people doing it so inefficient?
    * See also: numerics.sigmoid.inPlace(raw.values)
    * This is approximately 25% faster.
    */
  @inline
  final def fill[T](dst0: DenseVector[T])
                   (fn: => T)
  : Unit = ArrayEx.fill(
    dst0.data, dst0.offset, dst0.stride,
    dst0.length
  )(fn)

  @inline
  final def fill[T, U](dst0: DenseVector[T],
                       src1: DenseVector[U])
                      (fn: U => T)
  : Unit = {
    require(dst0.length == src1.length)
    ArrayEx.fill(
      dst0.data, dst0.offset, dst0.stride,
      src1.data, src1.offset, src1.stride,
      dst0.length
    )(fn)
  }

  @inline
  final def filter[T](dst0: SparseVector[T])
                     (predicate: T => Boolean)
  : Unit = dst0.array.filter(predicate)

  @inline
  final def foldLeft[T, U](src0: T,
                           src1: Vector[U])
                          (fn: (T, U) => T)
  : T = src1 match {
    case src1: DenseVector[U] =>
      foldLeft(
        src0,
        src1
      )(fn)
    case src1: SparseVector[U] =>
      foldLeft(
        src0,
        src1
      )(fn)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def foldLeft[T, U](src0: T,
                           src1: DenseVector[U])
                          (fn: (T, U) => T)
  : T = ArrayEx.foldLeft(
    src0,
    src1.data, src1.offset, src1.stride,
    src1.length
  )(fn)

  @inline
  final def foldLeft[T, U](src0: T,
                           src1: SparseVector[U])
                          (fn: (T, U) => T)
  : T = ArrayEx.foldLeft(
    src0,
    src1.array
  )(fn)

  @inline
  final def foldLeft[T, U, V](src0: T,
                              src1: DenseVector[U],
                              src2: Vector[V])
                             (fn: (T, U, V) => T)
  : T = src2 match {
    case src2: DenseVector[V] =>
      foldLeft(
        src0,
        src1,
        src2
      )(fn)
    case src2: SparseVector[V] =>
      foldLeft(
        src0,
        src1,
        src2
      )(fn)
    case _ =>
      throw new MatchError(src2)
  }

  @inline
  final def foldLeft[T, U, V](src0: T,
                              src1: DenseVector[U],
                              src2: DenseVector[V])
                             (fn: (T, U, V) => T)
  : T = {
    require(src1.length == src2.length)
    ArrayEx.foldLeft(
      src0,
      src1.data, src1.offset, src1.stride,
      src2.data, src2.offset, src2.stride,
      src1.length
    )(fn)
  }

  @inline
  final def foldLeft[T, U, V](src0: T,
                              src1: DenseVector[U],
                              src2: SparseVector[V])
                             (fn: (T, U, V) => T)
  : T = {
    require(src1.length == src2.length)
    ArrayEx.foldLeft(
      src0,
      src1.data, src1.offset, src1.stride,
      src2.array
    )(fn)
  }

  @inline
  final def foldLeft[T, U, V, W](src0: T,
                                 src1: DenseVector[U],
                                 src2: DenseVector[V],
                                 src3: DenseVector[W])
                                (fn: (T, U, V, W) => T)
  : T = {
    require(
      src1.length == src2.length &&
      src1.length == src3.length
    )
    ArrayEx.foldLeft(
      src0,
      src1.data, src1.offset, src1.stride,
      src2.data, src2.offset, src2.stride,
      src3.data, src3.offset, src3.stride,
      src1.length
    )(fn)
  }

  @inline
  final def foldLeftActive[T, U](src0: T,
                                 src1: Vector[U])
                                (fn: (T, U) => T)
  : T = src1 match {
    case src1: DenseVector[U] =>
      foldLeft(
        src0,
        src1
      )(fn)
    case src1: SparseVector[U] =>
      foldLeftActive(
        src0,
        src1
      )(fn)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def foldLeftActive[T, U](src0: T,
                                 src1: SparseVector[U])
                                (fn: (T, U) => T)
  : T = ArrayEx.foldLeftActive(
    src0,
    src1.array
  )(fn)

  @inline
  final def foldLeftEx[T, U, V](src0:  T,
                                src1: DenseVector[U],
                                src2: Vector[V])
                               (fn0: (T, U, V) => T, fn1: (T, U) => T)
  : T = src2 match {
    case src2: DenseVector[V] =>
      foldLeftEx(
        src0,
        src1,
        src2
      )(fn0, fn1)
    case src2: SparseVector[V] =>
      foldLeftEx(
        src0,
        src1,
        src2
      )(fn0, fn1)
    case _ =>
      throw new MatchError(src2)
  }

  @inline
  final def foldLeftEx[T, U, V](src0:  T,
                                src1: DenseVector[U],
                                src2: DenseVector[V])
                               (fn0: (T, U, V) => T, fn1: (T, U) => T)
  : T = foldLeft(
    src0,
    src1,
    src2
  )(fn0)

  @inline
  final def foldLeftEx[T, U, V](src0: T,
                                src1: DenseVector[U],
                                src2: SparseVector[V])
                               (fn0: (T, U, V) => T, fn1: (T, U) => T)
  : T = {
    require(src1.length == src2.length)
    ArrayEx.foldLeftEx(
      src0,
      src1.data, src1.offset, src1.stride,
      src2.array
    )(fn0, fn1)
  }

  @inline
  final def foreach[T](src0: Vector[T])
                      (fn: T => Unit)
  : Unit = src0 match {
    case src0: DenseVector[T] =>
      foreach(
        src0
      )(fn)
    case src0: SparseVector[T] =>
      foreach(
        src0
      )(fn)
    case _ =>
      throw new MatchError(src0)
  }

  // TODO: Fix submitted on GitHub for CanTransformValues: https://github.com/scalanlp/breeze/pull/420
  // TODO: Can remove this function if David updates built-in foreach accordingly.
  @inline
  final def foreach[T](src0: DenseVector[T])
                      (fn: T => Unit)
  : Unit = ArrayEx.foreach(
    src0.data, src0.offset, src0.stride,
    src0.length
  )(fn)

  @inline
  final def foreach[T](src0: SparseVector[T])
                      (fn: T => Unit)
  : Unit = ArrayEx.foreach(
    src0.array
  )(fn)

  @inline
  final def foreach[T, U](src0: DenseVector[T],
                          src1: Vector[U])
                         (fn: (T, U) => Unit)
  : Unit = src1 match {
    case src1: DenseVector[U] =>
      foreach(
        src0,
        src1
      )(fn)
    case src1: SparseVector[U] =>
      foreach(
        src0,
        src1
      )(fn)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def foreach[T, U](src0: DenseVector[T],
                          src1: DenseVector[U])
                         (fn: (T, U) => Unit)
  : Unit = {
    require(src0.length == src1.length)
    ArrayEx.foreach(
      src0.data, src0.offset, src0.stride,
      src1.data, src1.offset, src1.stride,
      src0.length
    )(fn)
  }

  @inline
  final def foreach[T, U](src0: DenseVector[T],
                          src1: SparseVector[U])
                         (fn: (T, U) => Unit)
  : Unit = foreachEx(
    src0,
    src1
  )(fn, fn(_, src1.default))

  @inline
  final def foreach[T, U, V](src0: DenseVector[T],
                             src1: DenseVector[U],
                             src2: DenseVector[V])
                            (fn: (T, U, V) => Unit)
  : Unit = {
    require(
      src0.length == src1.length &&
      src0.length == src2.length
    )
    ArrayEx.foreach(
      src0.data, src0.offset, src0.stride,
      src1.data, src1.offset, src1.stride,
      src2.data, src2.offset, src2.stride,
      src0.length
    )(fn)
  }

  @inline
  final def foreachActive[T](src0: Vector[T])
                            (fn: T => Unit)
  : Unit = src0 match {
    case src0: DenseVector[T] =>
      foreach(
        src0
      )(fn)
    case src0: SparseVector[T] =>
      foreachActive(
        src0
      )(fn)
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def foreachActive[T](src0: SparseVector[T])
                            (fn: T => Unit)
  : Unit = ArrayEx.foreachActive(
    src0.array
  )(fn)

  @inline
  final def foreachActivePair[T](src0: Vector[T])
                                (fn: (Int, T) => Unit)
  : Unit = src0 match {
    case src0: DenseVector[T] =>
      foreachPair(
        src0
      )(fn)
    case src0: SparseVector[T] =>
      foreachActivePair(
        src0
      )(fn)
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def foreachActivePair[T](src0: SparseVector[T])
                                (fn: (Int, T) => Unit)
  : Unit = ArrayEx.foreachActivePair(
    src0.array
  )(fn)

  @inline
  final def foreachEx[T, U](src0: DenseVector[T],
                            src1: Vector[U])
                           (fn0: (T, U) => Unit, fn1: T => Unit)
  : Unit = src1 match {
    case src1: DenseVector[U] =>
      foreach(
        src0,
        src1
      )(fn0)
    case src1: SparseVector[U] =>
      foreachEx(
        src0,
        src1
      )(fn0, fn1)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def foreachEx[T, U](src0: DenseVector[T],
                            src1: SparseVector[U])
                           (fn0: (T, U) => Unit, fn1: T => Unit)
  : Unit = {
    require(src0.length == src1.length)
    ArrayEx.foreachEx(
      src0.data, src0.offset, src0.stride,
      src1.array
    )(fn0, fn1)
  }

  @inline
  final def foreachPair[T](src0: DenseVector[T])
                          (fn: (Int, T) => Unit)
  : Unit = ArrayEx.foreachPair(
    src0.data, src0.offset, src0.stride,
    src0.length
  )(fn)

  @inline
  final def foreachPair[T, U](src0: DenseVector[T],
                              src1: Vector[U])
                             (fn: (Int, T, U) => Unit)
  : Unit = src1 match {
    case src1: DenseVector[U] =>
      foreachPair(
        src0,
        src1
      )(fn)
    case src1: SparseVector[U] =>
      foreachPair(
        src0,
        src1
      )(fn)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def foreachPair[T, U](src0: DenseVector[T],
                              src1: DenseVector[U])
                             (fn: (Int, T, U) => Unit)
  : Unit = {
    require(src0.length == src1.length)
    ArrayEx.foreachPair(
      src0.data, src0.offset, src0.stride,
      src1.data, src1.offset, src1.stride,
      src0.length
    )(fn)
  }

  @inline
  final def foreachPair[T, U](src0: DenseVector[T],
                              src1: SparseVector[U])
                             (fn: (Int, T, U) => Unit)
  : Unit = {
    require(src0.length == src1.length)
    ArrayEx.foreachPair(
      src0.data, src0.offset, src0.stride,
      src1.array
    )(fn)
  }

  @inline
  final def foreachPairEx[T, U](src0: DenseVector[T],
                                src1: Vector[U])
                               (fn0: (Int, T, U) => Unit, fn1: (Int, T) => Unit)
  : Unit = src1 match {
    case src1: DenseVector[U] =>
      foreachPair(
        src0,
        src1
      )(fn0)
    case src1: SparseVector[U] =>
      foreachPairEx(
        src0,
        src1
      )(fn0, fn1)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def foreachPairEx[T, U](src0: DenseVector[T],
                                src1: SparseVector[U])
                               (fn0: (Int, T, U) => Unit, fn1: (Int, T) => Unit)
  : Unit = {
    require(src0.length == src1.length)
    ArrayEx.foreachPairEx(
      src0.data, src0.offset, src0.stride,
      src1.array
    )(fn0, fn1)
  }

  @inline
  final def interleave[T](dst0: DenseVector[T],
                          src1: DenseVector[T],
                          src2: DenseVector[T])
                         (implicit tagT: ClassTag[T])
  : Unit = {
    require(
      dst0.length == src1.length * 2 &&
      dst0.length == src2.length * 2
    )
    ArrayEx.interleave(
      dst0.data, dst0.offset, dst0.stride,
      src1.data, src1.offset, src1.stride,
      src1.length,
      src2.data, src2.offset, src2.stride,
      src2.length
    )
  }

  @inline
  final def interleave[T](dst0: DenseVector[T],
                          src1: DenseVector[T],
                          src2: SparseVector[T])
                         (implicit tagT: ClassTag[T])
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length
    )
    ArrayEx.interleave(
      dst0.data, dst0.offset, dst0.stride,
      src1.data, src1.offset, src1.stride,
      src1.length,
      src2.array
    )
  }

  @inline
  final def interleave[T](dst0: DenseVector[T],
                          src1: SparseVector[T],
                          src2: DenseVector[T])
                         (implicit tagT: ClassTag[T])
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length
    )
    ArrayEx.interleave(
      dst0.data, dst0.offset, dst0.stride,
      src1.array,
      src2.data, src2.offset, src2.stride,
      src2.length
    )
  }

  @inline
  final def interleave[T](src0: SparseVector[T],
                          src1: SparseVector[T])
                         (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : SparseArray[T] = {
    require(src0.length == src1.length)
    ArrayEx.interleave(
      src0.array,
      src1.array
    )
  }

  @inline
  final def head[T](src0: DenseVector[T])
  : T = {
    require(src0.length > 0)
    src0.data(src0.offset)
  }

  @inline
  final def l1Norm(src0: DenseVector[Real],
                   epsilon: Double)
  : Real = ArrayEx.l1Norm(
    src0.data, src0.offset, src0.stride,
    src0.length,
    epsilon
  )

  @inline
  final def l2Norm(src0: DenseVector[Real],
                   epsilon: Double)
  : Real = ArrayEx.l2Norm(
    src0.data, src0.offset, src0.stride,
    src0.length,
    epsilon
  )

  @inline
  final def l2NormSq(src0: DenseVector[Real])
  : Real = ArrayEx.l2NormSq(
    src0.data, src0.offset, src0.stride,
    src0.length
  )

  @inline
  final def labelsToDense(noClasses: Int,
                          classNo:   Int)
  : DenseVector[Real] = labelsToDense(noClasses, classNo, Real.one)

  @inline
  final def labelsToDense(noClasses: Int,
                          classNo:   Int,
                          value:     Real)
  : DenseVector[Real] = new DenseVector(
    ArrayEx.labelsToArray(noClasses, classNo, value)
  )

  @inline
  final def labelsToDenseEx(noClasses: Int,
                            classNos:  Seq[Int])
  : DenseVector[Real] = labelsToDenseEx(noClasses, classNos, Real.one)

  @inline
  final def labelsToDenseEx(noClasses: Int,
                            classNos:  Seq[Int],
                            value:     Real)
  : DenseVector[Real] = new DenseVector(
    ArrayEx.labelsToArrayEx(noClasses, classNos, value)
  )

  @inline
  final def labelsToSparse(noClasses: Int,
                           classNo:   Int)
  : SparseVector[Real] = labelsToSparse(noClasses, classNo, Real.one)

  @inline
  final def labelsToSparse(noClasses: Int,
                           classNo:   Int,
                           value:     Real)
  : SparseVector[Real] = new SparseVector(
    ArrayEx.labelsToSparseArray(noClasses, classNo, value)
  )

  @inline
  final def labelsToSparseEx(noClasses: Int,
                             classNos:  Seq[Int])
  : SparseVector[Real] = labelsToSparseEx(noClasses, classNos, Real.one)

  @inline
  final def labelsToSparseEx(noClasses: Int,
                             classNos:  Seq[Int],
                             value:     Real)
  : SparseVector[Real] = new SparseVector(
    ArrayEx.labelsToSparseArrayEx(noClasses, classNos, value)
  )

  @inline
  final def last[T](src0: DenseVector[T])
  : T = src0.data(lastOffset(src0))

  @inline
  final def lastOffset[T](src0: DenseVector[T])
  : Int = {
    require(src0.length > 0)
    src0.offset + src0.stride * (src0.length - 1)
  }

  @inline
  final def lerp(src0: DenseVector[Real],
                 src1: DenseVector[Real],
                 t:    Real)
  : Unit = {
    require(src0.length == src1.length)
    ArrayEx.lerp(
      src0.data, src0.offset, src0.stride,
      src1.data, src1.offset, src1.stride,
      src0.length,
      t
    )
  }

  @inline
  final def map[T, U](src0: DenseVector[T])
                     (fn: T => U)
                     (implicit tagT: ClassTag[U])
  : Array[U] = ArrayEx.map(
    src0.data, src0.offset, src0.stride,
    src0.length
  )(fn)

  @inline
  final def mapActive[T, U](src0: Vector[T])
                           (fn: T => U)
                           (implicit tagU: ClassTag[U], zeroU: Zero[U])
  : Vector[U] = src0 match {
    case src0: DenseVector[T] =>
      val result = map(
        src0
      )(fn)
      new DenseVector(result)
    case src0: SparseVector[T] =>
      val result = mapActive(
        src0
      )(fn)
      new SparseVector(result)
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def mapActive[T, U](src0: SparseVector[T])
                           (fn: T => U)
                           (implicit tagU: ClassTag[U], zeroU: Zero[U])
  : SparseArray[U] = ArrayEx.mapActive(
    src0.array
  )(fn)

  /**
    * This variant allows index access.
    */
  @inline
  final def mapPairs[T, U](src0: DenseVector[T])
                          (fn: (Int, T) => U)
                          (implicit tagU: ClassTag[U])
  : Array[U] = ArrayEx.mapPairs(
    src0.data, src0.offset, src0.stride,
    src0.length
  )(fn)

  @inline
  final def max(src0: DenseVector[Real])
  : Real = ArrayEx.max(
    src0.data, src0.offset, src0.stride,
    src0.length
  )

  @inline
  final def maxIndex(src0: DenseVector[Real])
  : Int = ArrayEx.maxIndex(
    src0.data, src0.offset, src0.stride,
    src0.length
  )

  @inline
  final def mean(src0: DenseVector[Real])
  : Real = ArrayEx.mean(
    src0.data, src0.offset, src0.stride,
    src0.length
  )

  @inline
  final def median(src0: DenseVector[Real])
  : Real = breeze.stats.median(src0)

  @inline
  final def memoryUtilization[T](src0: SparseVector[T])
  : Real = ArrayEx.memoryUtilization(src0.array)

  @inline
  final def min(src0: DenseVector[Real])
  : Real = ArrayEx.min(
    src0.data, src0.offset, src0.stride,
    src0.length
  )

  @inline
  final def minIndex(src0: DenseVector[Real])
  : Int = breeze.linalg.argmin(src0)

  @inline
  final def multiConcatDenseH[T](vectors: Array[DenseVector[T]])
                                (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : DenseMatrix[T] = {
    if (vectors.isEmpty) {
      DenseMatrix.zeros[T](0, 0)
    }
    else {
      // TODO: Could be done faster.
      val result = DenseMatrix.zeros[T](vectors.head.length, vectors.length)
      ArrayEx.foreachPair(vectors)(
        (i, v) => result(::, i) := v
      )
      result
    }
  }

  @inline
  final def multiConcatDenseH[T](vectors: Seq[DenseVector[T]])
                                (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : DenseMatrix[T] = {
    if (vectors.isEmpty) {
      DenseMatrix.zeros[T](0, 0)
    }
    else {
      // TODO: Could be done faster.
      val result = DenseMatrix.zeros[T](vectors.head.length, vectors.length)
      SeqEx.foreachPair(vectors)(
        (i, v) => result(::, i) := v
      )
      result
    }
  }

  @inline
  final def multiConcatDenseV[T](vectors: Array[Transpose[DenseVector[T]]])
                                (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : DenseMatrix[T] = {
    if (vectors.isEmpty) {
      DenseMatrix.zeros[T](0, 0)
    }
    else {
      // TODO: Could be done faster.
      val result = DenseMatrix.zeros[T](
        vectors.length, vectors.head.inner.length
      )
      ArrayEx.foreachPair(vectors)(
        (i, v) => result(i, ::) := v
      )
      result
    }
  }

  @inline
  final def multiConcatDenseV[T](vectors: Seq[Transpose[DenseVector[T]]])
                                (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : DenseMatrix[T] = {
    if (vectors.isEmpty) {
      DenseMatrix.zeros[T](0, 0)
    }
    else {
      // TODO: Could be done faster.
      val result = DenseMatrix.zeros[T](
        vectors.length, vectors.head.inner.length
      )
      SeqEx.foreachPair(vectors)(
        (i, v) => result(i, ::) := v
      )
      result
    }
  }

  @inline
  final def multiConcatSparseH[T](cols: Array[SparseVector[T]])
                                 (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : CSCMatrix[T] = {
    if (cols.isEmpty) {
      CSCMatrix.zeros[T](0, 0)
    }
    else {
      val result = CSCMatrix.zeros[T](cols.head.length, cols.length)
      ArrayEx.foreachPair(cols)(
        (c, col) => foreachActivePair(col)(
          (r, v) => result.update(r, c, v)
        )
      )
      result
    }
  }

  @inline
  final def multiConcatSparseH[T](cols: Seq[SparseVector[T]])
                                 (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : CSCMatrix[T] = {
    if (cols.isEmpty) {
      CSCMatrix.zeros[T](0, 0)
    }
    else {
      val result = CSCMatrix.zeros[T](cols.head.length, cols.length)
      SeqEx.foreachPair(cols)(
        (c, col) => foreachActivePair(col)(
          (r, v) => result.update(r, c, v)
        )
      )
      result
    }
  }

  final def multiConcatSparseV[T](rows: Array[SparseVector[T]])
                                 (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : CSCMatrix[T] = {
    if (rows.isEmpty) {
      CSCMatrix.zeros[T](0, 0)
    }
    else {
      val result = CSCMatrix.zeros[T](rows.length, rows.head.length)
      ArrayEx.foreachPair(rows)(
        (r, row) => foreachActivePair(row)(
          result.update(r, _, _)
        )
      )
      result
    }
  }

  final def multiConcatSparseV[T](rows: Seq[SparseVector[T]])
                                 (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : CSCMatrix[T] = {
    if (rows.isEmpty) {
      CSCMatrix.zeros[T](0, 0)
    }
    else {
      val result = CSCMatrix.zeros[T](rows.length, rows.head.length)
      SeqEx.foreachPair(rows)(
        (r, row) => foreachActivePair(row)(
          result.update(r, _, _)
        )
      )
      result
    }
  }

  @inline
  final def multiply[T](dst0: DenseVector[Real],
                        src1:  Real)
  : Unit = ArrayEx.multiply(
    dst0.data, dst0.offset, dst0.stride,
    src1,
    dst0.length
  )

  @inline
  final def multiply[T](dst0: DenseVector[Real],
                        Src1: DenseVector[Real])
  : Unit = {
    require(dst0.length == Src1.length)
    ArrayEx.multiply(
      dst0.data, dst0.offset, dst0.stride,
      Src1.data, Src1.offset, Src1.stride,
      dst0.length
    )
  }

  @inline
  final def offsetAt[T](src0: DenseVector[T],
                        index: Int)
  : Int = src0.offset + src0.stride * index

  @inline
  final def offsets[T](src0: DenseVector[T])
  : Range = src0.offset until endOffset(src0) by src0.stride

  @inline
  final def populationStdDev(src0: DenseVector[Real],
                             epsilon: Double)
  : Real = ArrayEx.populationStdDev(
    src0.data, src0.offset, src0.stride,
    src0.length,
    epsilon
  )

  @inline
  final def populationStdDev(src0: DenseVector[Real],
                             mean1:   Real,
                             epsilon: Double)
  : Real = ArrayEx.populationStdDev(
    src0.data, src0.offset, src0.stride,
    mean1,
    src0.length,
    epsilon
  )

  @inline
  final def populationVariance(src0: DenseVector[Real])
  : Real = ArrayEx.populationVariance(
    src0.data, src0.offset, src0.stride,
    src0.length
  )

  @inline
  final def populationVariance(src0: DenseVector[Real],
                               mean1: Real)
  : Real = ArrayEx.populationVariance(
    src0.data, src0.offset, src0.stride,
    mean1,
    src0.length
  )

  @inline
  final def reduceLeft[T](src0: DenseVector[T])
                         (fn: (T, T) => T)
  : T = ArrayEx.reduceLeft(
    src0.data, src0.offset, src0.stride,
    src0.length
  )(fn)

  @inline
  final def reduceLeft[T](src0: SparseVector[T])
                         (fn: (T, T) => T)
                         (implicit zero: Zero[T])
  : T = ArrayEx.reduceLeft(
    src0.array
  )(fn)

  @inline
  final def reduceLeftActive[T](src0: SparseVector[T])
                               (fn: (T, T) => T)
  : T = ArrayEx.reduceLeftActive(
    src0.array
  )(fn)

  /**
    * Equivalent of Matlabs repmat function.
    *
    * @param cols Number of copies of the column-vector.
    * @return
    */
  @inline
  final def repeatH[T](src0: DenseVector[T], cols: Int)
                      (implicit classTag: ClassTag[T], zero: Zero[T])
  : DenseMatrix[T] = {
    val result = DenseMatrix.zeros[T](src0.length, cols)
    repeatH(src0, result)
    result
  }

  @inline
  final def repeatH[T](src0: DenseVector[T],
                       sink: DenseMatrix[T])
  : Unit = {
    val cols = sink.cols
    val rows = sink.rows

    require(!sink.isTranspose && rows == src0.length)

    if (cols > 0) {
      // Fetch frequently used values.
      val data1   = sink.data
      val stride1 = sink.majorStride
      val data0   = src0.data
      val stride0 = src0.stride

      if (stride0 == 1) {
        val offset0 = src0.offset
        var offset1 = sink.offset
        val end1    = sink.offset + stride1 * cols
        while (offset1 < end1) {
          ArrayEx.set(
            data1, offset1, 1,
            data0, offset0, 1,
            rows
          )
          offset1 += stride1
        }
      }
      else {
        // Copy first column.
        var off1 = sink.offset
        var off0 = src0.offset
        val end0 = src0.offset + stride0 * rows
        while (off0 != end0) {
          data1(off1) = data0(off0)
          off0 += stride0
          off1 += 1
        }

        // Copy remaining columns.
        val end1 = sink.offset + stride1 * cols
        while (off1 != end1) {
          ArrayEx.set(
            data1, off1,        1,
            data0, sink.offset, 1,
            rows
          )
          off1 += stride1
        }
      }
    }
  }

  @inline
  final def sampleStdDev(src0: DenseVector[Real],
                         epsilon: Double)
  : Real = ArrayEx.sampleStdDev(
    src0.data, src0.offset, src0.stride,
    src0.length,
    epsilon
  )

  @inline
  final def sampleStdDev(src0: DenseVector[Real],
                         mean1:   Real,
                         epsilon: Double)
  : Real = ArrayEx.sampleStdDev(
    src0.data, src0.offset, src0.stride,
    mean1,
    src0.length,
    epsilon
  )

  @inline
  final def sampleVariance(src0: DenseVector[Real])
  : Real = ArrayEx.sampleVariance(
    src0.data, src0.offset, src0.stride,
    src0.length
  )

  @inline
  final def sampleVariance(src0: DenseVector[Real],
                           mean1: Real)
  : Real = ArrayEx.sampleVariance(
    src0.data, src0.offset, src0.stride,
    mean1,
    src0.length
  )

  @inline
  final def sizeOf(src0: Vector[Real])
  : Long = src0 match {
    case src0: DenseVector[Real] =>
      sizeOf(src0)
    case src0: SparseVector[Real] =>
      sizeOf(src0)
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def sizeOf(src0: DenseVector[Real])
  : Long = 8L + 4L + 4L + 4L + 1L + ArrayEx.sizeOf(src0.data)

  @inline
  final def sizeOf(src0: SparseVector[Real])
  : Long = 8L + ArrayEx.sizeOf(src0.array)

  @inline
  final def sum(src0: DenseVector[Real])
  : Real = ArrayEx.sum(
    src0.data, src0.offset, src0.stride,
    src0.length
  )

  @inline
  final def tabulate[T](length: Int)
                       (fn: Int => T)
                       (implicit tagT: ClassTag[T])
  : DenseVector[T] = {
    val result = ArrayEx.tabulate(
      length
    )(fn)
    DenseVector(result)
  }

  @inline
  final def tabulate[T](dst0: DenseVector[T])
                       (fn: Int => T)
  : Unit = ArrayEx.tabulate(
    dst0.data, dst0.offset, dst0.stride,
    dst0.length
  )(fn)

  @inline
  final def tailor[T](dst0: SparseVector[T])
  : Unit = ArrayEx.tailor(dst0.array)

  @inline
  final def toArray[T](src0: SparseVector[T])
                      (implicit tagT: ClassTag[T])
  : Array[T] = ArrayEx.toArray(src0.array)

  @inline
  final def toDense[T](src0: SparseVector[T])
                      (implicit tagT: ClassTag[T])
  : DenseVector[T] = {
    val result = toArray(src0)
    DenseVector(result)
  }

  // TODO: Yet another workaround because breeze bug: https://github.com/scalanlp/breeze/issues/446
  @inline
  final def toMatrix[T](src0: SparseVector[T])
                       (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : CSCMatrix[T] = {
    // TODO: Could be done faster.
    val result = CSCMatrix.zeros[T](src0.length, 1)
    foreachActivePair(
      src0
    )((i, v) => result.update(i, 0, v))
    result
  }

  @inline
  final def toSparse[T](src0: DenseVector[T])
                       (predicate: T => Boolean)
                       (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : SparseVector[T] = {
    val result = ArrayEx.toSparseArray(
      src0.data, src0.offset, src0.stride,
      src0.length
    )(predicate)
    new SparseVector(result)
  }

  @inline
  final def toSparse[T](length: Int, values: Map[Int, T])
                       (implicit tagT: ClassTag[T], zeroT: Zero[T])
  : SparseVector[T] = {
    val result = SparseVector.zeros[T](length)
    result.reserve(values.size)
    values.foreach(
      kv => result.update(kv._1, kv._2)
    )
    result
  }

  /**
    * Other method quite slow... Why are the Breeze people doing it so inefficient?
    * See also: numerics.sigmoid.inPlace(raw.values)
    * This is approximately 25% faster.
    */
  // TODO: Fix submitted on GitHub for CanTransformValues: https://github.com/scalanlp/breeze/pull/420
  @inline
  final def transform[T](dst0: DenseVector[T])
                        (fn: T => T)
  : Unit = ArrayEx.transform(
    dst0.data, dst0.offset, dst0.stride,
    dst0.length
  )(fn)

  /**
    * A more advanced version of updateEach. What this basically does is a
    * zip.inPlace.
    */
  @inline
  final def transform[T, U](dst0: DenseVector[T],
                            src1: DenseVector[U])
                           (fn: (T, U) => T)
  : Unit = {
    require(dst0.length == src1.length)
    ArrayEx.transform(
      dst0.data, dst0.offset, dst0.stride,
      src1.data, src1.offset, src1.stride,
      dst0.length
    )(fn)
  }

  final def transform[T, U](dst0: DenseVector[T],
                            src1: SparseVector[U])
                           (fn: (T, U) => T)
  : Unit = {
    require(dst0.length == src1.length)
    ArrayEx.transform(
      dst0.data, dst0.offset, dst0.stride,
      src1.array
    )(fn)
  }

  @inline
  final def transform[T, U, V](dst0: DenseVector[T],
                               src1: DenseVector[U],
                               src2: DenseVector[V])
                              (fn: (T, U, V) => T)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length
    )
    ArrayEx.transform(
      dst0.data, dst0.offset, dst0.stride,
      src1.data, src1.offset, src1.stride,
      src2.data, src2.offset, src2.stride,
      dst0.length
    )(fn)
  }

  @inline
  final def transform[T, U, V, W](dst0: DenseVector[T],
                                  src1: DenseVector[U],
                                  src2: DenseVector[V],
                                  src3: DenseVector[W])
                                 (fn: (T, U, V, W) => T)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length &&
      dst0.length == src3.length
    )
    ArrayEx.transform(
      dst0.data, dst0.offset, dst0.stride,
      src1.data, src1.offset, src1.stride,
      src2.data, src2.offset, src2.stride,
      src3.data, src3.offset, src3.stride,
      dst0.length
    )(fn)
  }

  @inline
  final def transform[T, U, V, W, X](dst0: DenseVector[T],
                                     src1: DenseVector[U],
                                     src2: DenseVector[V],
                                     src3: DenseVector[W],
                                     src4: DenseVector[X])
                                    (fn: (T, U, V, W, X) => T)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length &&
      dst0.length == src3.length &&
      dst0.length == src4.length
    )
    ArrayEx.transform(
      dst0.data, dst0.offset, dst0.stride,
      src1.data, src1.offset, src1.stride,
      src2.data, src2.offset, src2.stride,
      src3.data, src3.offset, src3.stride,
      src4.data, src4.offset, src4.stride,
      dst0.length
    )(fn)
  }

  @inline
  final def transform[T, U, V, W, X, Y](dst0: DenseVector[T],
                                        src1: DenseVector[U],
                                        src2: DenseVector[V],
                                        src3: DenseVector[W],
                                        src4: DenseVector[X],
                                        src5: DenseVector[Y])
                                       (fn: (T, U, V, W, X, Y) => T)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length &&
      dst0.length == src3.length &&
      dst0.length == src4.length &&
      dst0.length == src5.length
    )
    ArrayEx.transform(
      dst0.data, dst0.offset, dst0.stride,
      src1.data, src1.offset, src1.stride,
      src2.data, src2.offset, src2.stride,
      src3.data, src3.offset, src3.stride,
      src4.data, src4.offset, src4.stride,
      src5.data, src5.offset, src5.stride,
      dst0.length
    )(fn)
  }

  @inline
  final def transformActive[T](dst0: Vector[T])
                              (fn: T => T)
  : Unit = dst0 match {
    case dst0: DenseVector[T] =>
      transform(
        dst0
      )(fn)
    case dst0: SparseVector[T] =>
      transformActive(
        dst0
      )(fn)
    case _ =>
      throw new MatchError(dst0)
  }

  @inline
  final def transformActive[T](dst0: SparseVector[T])
                              (fn: T => T)
  : Unit = ArrayEx.transformActive(dst0.array)(fn)

  @inline
  final def transformEx[T, U](dst0: DenseVector[T],
                              src1: Vector[U])
                             (fn0: (T, U) => T, fn1: T => T)
  : Unit = src1 match {
    case src1: DenseVector[U] =>
      transform(
        dst0,
        src1
      )(fn0)
    case src1: SparseVector[U] =>
      transformEx(
        dst0,
        src1
      )(fn0, fn1)
    case _ =>
      throw new MatchError(src1)
  }

  final def transformEx[T, U](dst0: DenseVector[T],
                              src1: SparseVector[U])
                             (fn0: (T, U) => T, fn1: T => T)
  : Unit = {
    require(dst0.length == src1.length)
    ArrayEx.transformEx(
      dst0.data, dst0.offset, dst0.stride,
      src1.array
    )(fn0, fn1)
  }

  /**
    * This variant allows index access.
    */
  @inline
  final def transformPairs[T](dst0: DenseVector[T])
                             (fn: (Int, T) => T)
  : Unit = ArrayEx.transformPairs(
    dst0.data, dst0.offset, dst0.stride,
    dst0.length
  )(fn)

  @inline
  final def transformPairs[T, U](dst0: DenseVector[T],
                                 src1: DenseVector[U])
                                (fn: (Int, T, U) => T)
  : Unit = {
    require(dst0.length == src1.length)
    ArrayEx.transformPairs(
      dst0.data, dst0.offset, dst0.stride,
      src1.data, src1.offset, src1.stride,
      dst0.length
    )(fn)
  }

  @inline
  final def transformPairs[T, U, V](dst0: DenseVector[T],
                                    src1: DenseVector[U],
                                    src2: DenseVector[V])
                                   (fn: (Int, T, U, V) => T)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length
    )
    ArrayEx.transformPairs(
      dst0.data, dst0.offset, dst0.stride,
      src1.data, src1.offset, src1.stride,
      src2.data, src2.offset, src2.stride,
      dst0.length
    )(fn)
  }

  @inline
  final def transformPairs[T, U, V, W](dst0: DenseVector[T],
                                       src1: DenseVector[U],
                                       src2: DenseVector[V],
                                       src3: DenseVector[W])
                                      (fn: (Int, T, U, V, W) => T)
  : Unit = {
    require(
      dst0.length == src1.length &&
      dst0.length == src2.length &&
      dst0.length == src3.length
    )
    ArrayEx.transformPairs(
      dst0.data, dst0.offset, dst0.stride,
      src1.data, src1.offset, src1.stride,
      src2.data, src2.offset, src2.stride,
      src3.data, src3.offset, src3.stride,
      dst0.length
    )(fn)
  }

  @inline
  final def utilization[T](src0: Vector[T])
                          (predicate: T => Boolean)
  : Real = src0 match {
    case src0: DenseVector[T] =>
      utilization(
        src0
      )(predicate)
    case src0: SparseVector[T] =>
      utilization(
        src0
      )(predicate)
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def utilization[T](src0: DenseVector[T])
                          (predicate: T => Boolean)
  : Real = count(
    src0
  )(predicate) / Real(src0.length)

  @inline
  final def utilization[T](src0: SparseVector[T])
                          (predicate: T => Boolean)
  : Real = count(
    src0
  )(predicate) / Real(src0.length)

  @inline
  final def utilizationApprox[T](src0: Vector[T],
                                 rng:       PseudoRNG,
                                 noSamples: Int)
                                (predicate: T => Boolean)
  : Real = src0 match {
    case src0: DenseVector[T] =>
      utilizationApprox(
        src0,
        rng,
        noSamples
      )(predicate)
    case src0: SparseVector[T] =>
      utilizationApprox(
        src0,
        rng,
        noSamples
      )(predicate)
    case _ =>
      throw new MatchError(src0)
  }

  @inline
  final def utilizationApprox[T](src0: DenseVector[T],
                                 rng:       PseudoRNG,
                                 noSamples: Int)
                                (predicate: T => Boolean)
  : Real = {
    val result = countApprox(
      src0,
      rng,
      noSamples
    )(predicate)
    result / Real(src0.length)
  }

  @inline
  final def utilizationApprox[T](src0: SparseVector[T],
                                 rng:       PseudoRNG,
                                 noSamples: Int)
                                (predicate: T => Boolean)
  : Real = ArrayEx.utilizationApprox(
    src0.array,
    rng,
    noSamples
  )(predicate)

  @inline
  final def zip[T, U, V](src0: DenseVector[T],
                         src1: Vector[U])
                        (fn: (T, U) => V)
                        (implicit tagV: ClassTag[V])
  : Array[V] = src1 match {
    case src1: DenseVector[U] =>
      zip(
        src0,
        src1
      )(fn)
    case src1: SparseVector[U] =>
      zip(
        src0,
        src1
      )(fn)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def zip[T, U, V](src0: DenseVector[T],
                         src1: DenseVector[U])
                        (fn: (T, U) => V)
                        (implicit tagV: ClassTag[V])
  : Array[V] = {
    require(src0.length == src1.length)
    ArrayEx.zip(
      src0.data, src0.offset, src0.stride,
      src1.data, src1.offset, src1.stride,
      src0.length
    )(fn)
  }

  @inline
  final def zip[T, U, V](src0: DenseVector[T],
                         src1: SparseVector[U])
                        (fn: (T, U) => V)
                        (implicit tagV: ClassTag[V])
  : Array[V] = {
    require(src0.length == src1.length)
    ArrayEx.zip(
      src0.data, src0.offset, src0.stride,
      src1.array
    )(fn)
  }

  @inline
  final def zip[T, U, V](src0: SparseVector[T],
                         src1: DenseVector[U])
                        (fn: (T, U) => V)
                        (implicit tagV: ClassTag[V])
  : Array[V] = zip(
    src1,
    src0
  )((a, b) => fn(b, a))

  @inline
  final def zip[T, U, V](src0: SparseVector[T],
                         src1: SparseVector[U])
                        (fn: (T, U) => V)
                        (implicit tagV: ClassTag[V], zeroV: Zero[V])
  : SparseArray[V] = ArrayEx.zip(
    src0.array,
    src1.array
  )(fn)

  @inline
  final def zip[T, U, V, W](src0: DenseVector[T],
                            src1: DenseVector[U],
                            src2: DenseVector[V])
                           (fn: (T, U, V) => W)
                           (implicit tagW: ClassTag[W])
  : Array[W] = {
    require(
      src0.length == src1.length &&
      src0.length == src2.length
    )
    ArrayEx.zip(
      src0.data, src0.offset, src0.stride,
      src1.data, src1.offset, src1.stride,
      src2.data, src2.offset, src2.stride,
      src0.length
    )(fn)
  }

  @inline
  final def zipEx[T, U, V](src0: DenseVector[T],
                           src1: Vector[U])
                          (fn0: (T, U) => V, fn1: T => V)
                          (implicit tagV: ClassTag[V])
  : Array[V] = src1 match {
    case src1: DenseVector[U] =>
      zip(
        src0,
        src1
      )(fn0)
    case src1: SparseVector[U] =>
      zipEx(
        src0,
        src1
      )(fn0, fn1)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def zipEx[T, U, V](src0: DenseVector[T],
                           src1: SparseVector[U])
                          (fn0: (T, U) => V, fn1: T => V)
                          (implicit tagV: ClassTag[V])
  : Array[V] = {
    require(src0.length == src1.length)
    ArrayEx.zipEx(
      src0.data, src0.offset, src0.stride,
      src1.array
    )(fn0, fn1)
  }

  @inline
  final def zipEx[T, U, V](src0: SparseVector[T],
                           src1: Vector[U])
                          (fn0: (T, U) => V, fn1: T => V, fn2: U => V)
                          (implicit tagV: ClassTag[V], zeroV: Zero[V])
  : Vector[V] = src1 match {
    case src1: DenseVector[U] =>
      val result = zipEx(
        src0,
        src1
      )(fn0, fn2)
      new DenseVector(result)
    case src1: SparseVector[U] =>
      val result = zipEx(
        src0,
        src1
      )(fn0, fn1, fn2)
      new SparseVector(result)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def zipEx[T, U, V](src0: SparseVector[T],
                           src1: DenseVector[U])
                          (fn0: (T, U) => V, fn1: U => V)
                          (implicit tagV: ClassTag[V])
  : Array[V] = zipEx(
    src1,
    src0
  )((a, b) => fn0(b, a), fn1)

  @inline
  final def zipEx[T, U, V](src0: SparseVector[T],
                           src1: SparseVector[U])
                          (fn0: (T, U) => V, fn1: T => V, fn2: U => V)
                          (implicit tagV: ClassTag[V], zeroV: Zero[V])
  : SparseArray[V] = ArrayEx.zipEx(
    src0.array,
    src1.array
  )(fn0, fn1, fn2)

  @inline
  final def zipPairs[T, U, V](src0: DenseVector[T],
                              src1: Vector[U])
                             (fn: (Int, T, U) => V)
                             (implicit tagV: ClassTag[V])
  : Array[V] = src1 match {
    case src1: DenseVector[U] =>
      zipPairs(
        src0,
        src1
      )(fn)
    case src1: SparseVector[U] =>
      zipPairs(
        src0,
        src1
      )(fn)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def zipPairs[T, U, V](src0: DenseVector[T],
                              src1: DenseVector[U])
                             (fn: (Int, T, U) => V)
                             (implicit tagV: ClassTag[V])
  : Array[V] = {
    require(src0.length == src1.length)
    ArrayEx.zipPairs(
      src0.data, src0.offset, src0.stride,
      src1.data, src1.offset, src1.stride,
      src0.length
    )(fn)
  }

  @inline
  final def zipPairs[T, U, V](src0: DenseVector[T],
                              src1: SparseVector[U])
                             (fn: (Int, T, U) => V)
                             (implicit tagV: ClassTag[V])
  : Array[V] = {
    require(src0.length == src1.length)
    ArrayEx.zipPairs(
      src0.data, src0.offset, src0.stride,
      src1.array
    )(fn)
  }

  @inline
  final def zipPairs[T, U, V](src0: SparseVector[T],
                              src1: DenseVector[U])
                             (fn: (Int, T, U) => V)
                             (implicit tagV: ClassTag[V])
  : Array[V] = zipPairs(
    src1,
    src0
  )((i, a, b) => fn(i, b, a))

  @inline
  final def zipPairs[T, U, V](src0: SparseVector[T],
                              src1: SparseVector[U])
                             (fn: (Int, T, U) => V)
                             (implicit tagV: ClassTag[V], zeroT: Zero[T], zeroU: Zero[U], zeroV: Zero[V])
  : SparseArray[V] = zipPairsEx(
    src0,
    src1
  )(fn, fn(_, _, zeroU.zero), fn(_, zeroT.zero, _))

  @inline
  final def zipPairs[T, U, V, W](src0: DenseVector[T],
                                 src1: DenseVector[U],
                                 src2: DenseVector[V])
                                (fn: (Int, T, U, V) => W)
                                (implicit tagW: ClassTag[W])
  : Array[W] = {
    require(
      src0.length == src1.length &&
      src0.length == src2.length
    )
    ArrayEx.zipPairs(
      src0.data, src0.offset, src0.stride,
      src1.data, src1.offset, src1.stride,
      src2.data, src2.offset, src2.stride,
      src0.length
    )(fn)
  }

  @inline
  final def zipPairsEx[T, U, V](src0: DenseVector[T],
                                src1: Vector[U])
                               (fn0: (Int, T, U) => V, fn1: (Int, T) => V)
                               (implicit tagV: ClassTag[V])
  : Array[V] = src1 match {
    case src1: DenseVector[U] =>
      zipPairs(
        src0,
        src1
      )(fn0)
    case src1: SparseVector[U] =>
      zipPairsEx(
        src0,
        src1
      )(fn0, fn1)
    case _ =>
      throw new MatchError(src1)
  }

  @inline
  final def zipPairsEx[T, U, V](src0: DenseVector[T],
                                src1: SparseVector[U])
                               (fn0: (Int, T, U) => V, fn1: (Int, T) => V)
                               (implicit tagV: ClassTag[V])
  : Array[V] = {
    require(src0.length == src1.length)
    ArrayEx.zipPairsEx(
      src0.data, src0.offset, src0.stride,
      src1.array
    )(fn0, fn1)
  }

  @inline
  final def zipPairsEx[T, U, V](src0: SparseVector[T],
                                src1: DenseVector[U])
                               (fn0: (Int, T, U) => V, fn1: (Int, U) => V)
                               (implicit tagV: ClassTag[V])
  : Array[V] = zipPairsEx(
    src1,
    src0
  )((i, a, b) => fn0(i, b, a), fn1)

  @inline
  final def zipPairsEx[T, U, V](src0: SparseVector[T],
                                src1: SparseVector[U])
                               (fn0: (Int, T, U) => V,
                                fn1: (Int, T) => V,
                                fn2: (Int, U) => V)
                               (implicit tagV: ClassTag[V], zeroV: Zero[V])
  : SparseArray[V] = ArrayEx.zipPairsEx(
    src0.array,
    src1.array
  )(fn0, fn1, fn2)

}
