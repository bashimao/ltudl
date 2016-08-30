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

import breeze.linalg.{DenseVector, Matrix, DenseMatrix, CSCMatrix}
import breeze.util.TopK
import scala.collection._

/**
  * Activations for a mini-batch.
  * All activation types inherit from this.
  */
// TODO: Change hashCode computation in a way that makes sure that different tensors with equivalent content have the same hashCode.
abstract class Tensor
  extends Closable
    with Equatable
    with Copyable
    with JsonSerializable {

  def dependsOn(other: Tensor)
  : Boolean = this eq other

  final override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Tensor]

  override def copy
  : Tensor

  def createSibling()
  : Tensor

  def createSibling(newLayout: TensorLayout)
  : Tensor

  def createSiblingAndClear()
  : Tensor

  def createSiblingAndClear(newLayout: TensorLayout)
  : Tensor


  // ---------------------------------------------------------------------------
  //    Data access related.
  // ---------------------------------------------------------------------------
  def platform
  : Platform

  def layout
  : TensorLayout

  def values
  : Array[Real]

  /**
    * @return A matrix containing the values. Might be a copy or not.
    */
  def valuesMatrix
  : DenseMatrix[Real]

  def valuesMatrixEx
  : Matrix[Real] with Serializable

  def get(valueNo: Int)
  : Real
  final def get(sampleNo: Int, valueNo: Int)
  : Real = {
    val index = layout.offsetFor(sampleNo, valueNo)
    get(index)
  }

  final def get(result: Array[Real])
  : Unit = get(result, 0, 1)

  def get(result: Array[Real], offset: Int, stride: Int)
  : Unit

  final def get(result: DenseVector[Real])
  : Unit = get(result.data, result.offset, result.stride)

  final def get(result: DenseMatrix[Real])
  : Unit = {
    require(!result.isTranspose && result.rows == result.majorStride)
    get(result.data, result.offset, 1)
  }

  def put(valueNo: Int, value: Real)
  : Unit

  final def put(sampleNo: Int, valueNo: Int, value: Real)
  : Unit = {
    val index = layout.offsetFor(sampleNo, valueNo)
    put(index, value)
  }

  final def put(array: Array[Real])
  : Unit = {
    require(array.length == layout.noValues)
    put(array, 0, 1)
  }

  def put(array: Array[Real], offset: Int, stride: Int)
  : Unit

  final def put(vector: DenseVector[Real])
  : Unit = {
    require(vector.length == layout.noValues)
    put(vector.data, vector.offset, vector.stride)
  }

  final def put(matrix: DenseMatrix[Real])
  : Unit = {
    val tmp = MatrixEx.asOrToArray(matrix)
    put(tmp, 0, 1)
  }

  def clear()
  : Unit

  final def fill(distribution: Distribution[Real])
  : Unit = fill(distribution.sample, distribution.isThreadSafe)

  def fill(fn: () => Real, threadSafe: Boolean)
  : Unit


  // ---------------------------------------------------------------------------
  //    Basic operations.
  // ---------------------------------------------------------------------------
  def reshape(size: Size)
  : Tensor

  def apply(index: Int)
  : Tensor

  def apply(indices: Range)
  : Tensor

  def splitSamples
  : Array[Tensor]

  def concat(other: Tensor)
  : Tensor

  def concat(other0: Tensor, others: Tensor*)
  : Tensor

  def concat[T <: Tensor](others: Array[T])
  : Tensor

  final def :=(value: Real)
  : Unit = {
    if (value == Real.zero) {
      clear()
    }
    else {
      doSet(value)
    }
  }

  protected def doSet(value: Real)
  : Unit

  final def :=(other: Tensor)
  : Unit = {
    require(layout == other.layout)
    doSet(other)
  }

  protected def doSet(other: Tensor)
  : Unit

  final def set(other: Tensor, beta: Real)
  : Unit = {
    require(layout == other.layout)
    beta match {
      case Real.zero =>
        clear()
      case Real.one =>
        doSet(other)
      case _ =>
        doSet(other, beta)
    }
  }

  protected def doSet(other: Tensor, beta: Real)
  : Unit

  final def set(other0: Tensor, other1: Tensor)
  : Unit = {
    val l = layout
    require(l == other0.layout && l == other1.layout)
    doSet(other0, other1)
  }

  protected def doSet(other0: Tensor, other1: Tensor)
  : Unit

  final def set(other0: Tensor, other1: Tensor, beta: Real)
  : Unit = {
    val l = layout
    require(l == other0.layout && l == other1.layout)
    beta match {
      case Real.zero =>
        clear()
      case Real.one =>
        doSet(other0, other1)
      case _ =>
        doSet(other0, other1, beta)
    }
  }

  protected def doSet(other0: Tensor, other1: Tensor, beta: Real)
  : Unit

  final def :=(other: Array[Real])
  : Unit = {
    require(other.length == layout.noValues)
    put(other, 0, 1)
  }

  final def :=(other: DenseVector[Real])
  : Unit = {
    require(other.length == layout.noValues)
    put(other.data, other.offset, other.stride)
  }

  final def :=(other: DenseMatrix[Real])
  : Unit = :=(MatrixEx.asOrToArray(other))

  def +(value: Real)
  : Tensor

  def +(other: Tensor)
  : Tensor

  final def +=(value: Real)
  : Unit = {
    if (value != Real.zero) {
      doAdd(value)
    }
  }

  protected def doAdd(value: Real)
  : Unit

  final def +=(other: Tensor)
  : Unit = {
    require(layout == other.layout)
    doAdd(other)
  }

  protected def doAdd(other: Tensor)
  : Unit

  final def add(alpha: Real,
                other: Tensor)
  : Unit = {
    require(layout == other.layout)
    alpha match {
      case Real.zero =>
        doSet(other)
      case Real.one =>
        doAdd(other)
      case _ =>
        doAdd(alpha, other)
    }
  }

  protected def doAdd(alpha: Real,
                      other: Tensor)
  : Unit

  final def add(other: Tensor, beta: Real)
  : Unit = {
    require(layout == other.layout)
    beta match {
      case Real.zero =>
      case Real.one =>
        doAdd(other)
      case _ =>
        doAdd(other, beta)
    }
  }

  protected def doAdd(other: Tensor, beta: Real)
  : Unit

  final def add(alpha: Real,
                other: Tensor, beta: Real)
  : Unit = {
    require(layout == other.layout)
    alpha match {
      case Real.zero =>
        doSet(other, beta)
      case Real.one =>
        doAdd(other, beta)
      case _ =>
        beta match {
          case Real.zero =>
            doMultiply(alpha)
          case Real.one =>
            doAdd(alpha, other)
          case _ =>
            doAdd(alpha, other, beta)
        }
    }
  }

  protected def doAdd(alpha: Real,
                      other: Tensor, beta: Real)
  : Unit

  final def add(other0: Tensor, other1: Tensor)
  : Unit = {
    val l = layout
    require(l == other0.layout && l == other1.layout)
    doAdd(other0, other1)
  }

  protected def doAdd(other0: Tensor, other1: Tensor)
  : Unit

  final def add(alpha:  Real,
                other0: Tensor, other1: Tensor)
  : Unit = {
    val l = layout
    require(l == other0.layout && l == other1.layout)
    alpha match {
      case Real.zero =>
        doSet(other0, other1)
      case Real.one =>
        doAdd(other0, other1)
      case _ =>
        doAdd(alpha, other0, other1)
    }
  }

  protected def doAdd(alpha:  Real,
                      other0: Tensor, other1: Tensor)
  : Unit

  final def add(other0: Tensor, other1: Tensor, beta: Real)
  : Unit = {
    val l = layout
    require(l == other0.layout && l == other1.layout)
    beta match {
      case Real.zero =>
      case Real.one =>
        doAdd(other0, other1)
      case _ =>
        doAdd(other0, other1, beta)
    }
  }

  protected def doAdd(other0: Tensor, other1: Tensor, beta: Real)
  : Unit

  final def add(alpha:  Real,
                other0: Tensor, other1: Tensor, beta: Real)
  : Unit = {
    val l = layout
    require(l == other0.layout && l == other1.layout)
    alpha match {
      case Real.zero =>
        doSet(other0, other1, beta)
      case Real.one =>
        doAdd(other0, other1, beta)
      case _ =>
        beta match {
          case Real.zero =>
            doMultiply(alpha)
          case Real.one =>
            doAdd(alpha, other0, other1)
          case _ =>
            doAdd(alpha, other0, other1, beta)
        }
    }
  }

  protected def doAdd(alpha:  Real,
                      other0: Tensor, other1: Tensor, beta: Real)
  : Unit

  def unary_-()
  : Tensor

  def -(other: Tensor)
  : Tensor

  final def -=(other: Tensor)
  : Unit = {
    require(layout == other.layout)
    doSubtract(other)
  }

  protected def doSubtract(other: Tensor)
  : Unit

  /**
    * unary_-, in place.
    */
  def subtractR(value: Real)
  : Unit

  def *(value: Real)
  : Tensor

  def :*(other: Tensor)
  : Tensor

  final def *=(value: Real)
  : Unit = {
    value match {
      case Real.zero =>
        clear()
      case Real.one =>
      case _ =>
        doMultiply(value)
    }
  }

  protected def doMultiply(value: Real)
  : Unit

  final def :*=(other: Tensor)
  : Unit = {
    if (this eq other) {
      sqr()
    }
    else {
      require(layout == other.layout)
      doMultiply(other)
    }
  }

  protected def doMultiply(other: Tensor)
  : Unit

  final def multiply(other: Tensor, beta: Real)
  : Unit = {
    require(layout == other.layout)
    beta match {
      case Real.zero =>
        clear()
      case Real.one =>
        doMultiply(other)
      case _ =>
        doMultiply(other, beta)
    }
  }

  protected def doMultiply(other: Tensor, beta: Real)
  : Unit

  def :/(other: Tensor)
  : Tensor

  final def :/=(other: Tensor)
  : Unit = {
    require(layout == other.layout)
    if (this eq other) {
      doSet(Real.one)
    }
    else {
      doDivide(other)
    }
  }

  protected def doDivide(other: Tensor)
  : Unit

  final def divide(epsilon0: Real,
                   other:    Tensor)
  : Unit = {
    require(layout == other.layout)
    if (epsilon0 == Real.zero) {
      doDivide(other)
    }
    else {
      doDivide(epsilon0, other)
    }
  }

  protected def doDivide(epsilon0: Real,
                         other:    Tensor)
  : Unit

  final def divide(other: Tensor, epsilon1: Real)
  : Unit = {
    require(layout == other.layout)
    if (epsilon1 == Real.zero) {
      doDivide(other)
    }
    else {
      doDivide(other, epsilon1)
    }
  }

  protected def doDivide(other: Tensor, epsilon1: Real)
  : Unit

  final def divide(epsilon0: Real,
                   other:    Tensor, epsilon1: Real)
  : Unit = {
    require(layout == other.layout)
    if (epsilon0 == Real.zero) {
      if (epsilon1 == Real.zero) {
        doDivide(other)
      }
      else {
        doDivide(other, epsilon1)
      }
    }
    else {
      if (epsilon1 == Real.zero) {
        doDivide(epsilon0, other)
      }
      else {
        doDivide(epsilon0, other, epsilon1)
      }
    }
  }

  protected def doDivide(epsilon0: Real,
                         other:    Tensor, epsilon1: Real)
  : Unit

  final def divideR(value: Real)
  : Unit = {
    if (value == Real.zero) {
      clear()
    }
    else {
      doDivideR(value)
    }
  }

  protected def doDivideR(value: Real)
  : Unit

  /**
    * Compute the reciprocal.
    */
  def reciprocal()
  : Tensor

  final def dot(other: Tensor)
  : Real = {
    require(layout == other.layout)
    doDot(other)
  }

  protected def doDot(other: Tensor)
  : Real

  final def lerp(other: Tensor, t: Real)
  : Unit = {
    require(layout == other.layout)
    t match {
      case Real.zero =>
      case Real.one =>
        doSet(other)
      case _ =>
        doLerp(other, t)
    }
  }

  protected def doLerp(other: Tensor, t: Real)
  : Unit

  final def lerp(other0: Tensor, other1: Tensor, t: Real)
  : Unit = {
    val l = layout
    require(l == other0.layout && l == other1.layout)
    t match {
      case Real.zero =>
      case Real.one =>
        doSet(other0, other1)
      case _ =>
        doLerp(other0, other1, t)
    }
  }

  protected def doLerp(other0: Tensor, other1: Tensor, t: Real)
  : Unit


  // ---------------------------------------------------------------------------
  //    Fancy operations.
  // ---------------------------------------------------------------------------
  def abs()
  : Unit

  final def approximateMean(rng: PseudoRNG)
  : Mean = approximateMean(rng, 1000)

  def approximateMean(rng:          PseudoRNG,
                      noSamplesMax: Int)
  : Mean

  final def approximateMeanAndVariance(rng: PseudoRNG)
  : MeanAndVariance = approximateMeanAndVariance(rng, 1000)

  def approximateMeanAndVariance(rng:          PseudoRNG,
                                 noSamplesMax: Int)
  : MeanAndVariance

  def l1Norm(epsilon: Double)
  : Real

  def l2Norm(epsilon: Double)
  : Real

  def l2NormSq
  : Real

  def max()
  : Real

  /**
    * Compare values and select max.
    */
  def max(other: Tensor)
  : Unit

  def maxAbs()
  : Real

  //def maxIndex: Int

  /**
    * Compares values without sign and chooses the larger value (signed!).
    * y = if (|y| > |x|) y else x
    */
  final def maxByAbs(other: Tensor)
  : Unit = {
    require(layout == other.layout)
    doMaxByAbs(other)
  }

  protected def doMaxByAbs(other: Tensor)
  : Unit

  def mean
  : Real

  def min()
  : Real

  /**
    * Compare values and select min.
    */
  def min(other: Tensor)
  : Unit

  //def minIndex: Int

  /**
    * Note that this works different from signum!
    */
  def sign()
  : Unit

  final def stdDev()
  : Real = stdDev(Real.zero)

  def stdDev(epsilon: Double)
  : Real

  def sqr()
  : Unit

  def sqrt()
  : Unit

  def sum
  : Real


  // ---------------------------------------------------------------------------
  //    Slicing.
  // ---------------------------------------------------------------------------
  /**
    * aaa ++ bbb = aaabbb
    */
  def ++(other: Tensor)
  : Tensor

  /**
    * aaa :++ bbb = ababab
    */
  def :++(other: Tensor)
  : Tensor

  def slice(tuple0: Int, noTuples: Int)
  : Tensor

  def slice(tuple0: Int, size: Size)
  : Tensor

  /**
    * Extracts a part of the tensor forming a new tensor. Whether this
    * copies the data or references the old memory area is not defined.
    * So you should not rely on either behavior.
    */
  final def slice(tuple0: Int, result: Tensor)
  : Unit = {
    val size        = layout.size
    val sliceLayout = result.layout
    val sliceSize   = sliceLayout.size
    require(
      tuple0                       >= 0               &&
      tuple0 + sliceSize.noTuples  <= size.noTuples   &&
      sliceSize.noChannels         == size.noChannels &&
      sliceLayout.noSamples        == layout.noSamples
    )
    doSlice(tuple0, result)
  }

  final def slice(result: Seq[Tensor])
  : Unit = {
    result.foldLeft(0)((tuple0, result) => {
      slice(tuple0, result)
      tuple0 + result.layout.size.noValues
    })
  }

  protected def doSlice(tuple0: Int,
                        result: Tensor)
  : Unit

  def sliceChannels(channel0: Int, noChannels: Int)
  : Tensor

  def sliceChannels(channel0: Int, size: Size)
  : Tensor

  final def sliceChannels(channel0: Int, result: Tensor)
  : Unit = {
    val size        = layout.size
    val sliceLayout = result.layout
    val sliceSize   = sliceLayout.size
    require(
      channel0                        >= 0               &&
      channel0 + sliceSize.noChannels <= size.noChannels &&
      sliceSize.noTuples              == size.noTuples   &&
      sliceLayout.noSamples           == layout.noSamples
    )
    doSliceChannels(channel0, result)
  }

  final def sliceChannels(result: Seq[Tensor])
  : Unit = {
    result.foldLeft(0)((channel0, result) => {
      sliceChannels(channel0, result)
      channel0 + result.layout.size.noChannels
    })
  }

  protected def doSliceChannels(channel0: Int,
                                result:   Tensor)
  : Unit


  // ---------------------------------------------------------------------------
  //    Conversion & extraction methods.
  // ---------------------------------------------------------------------------
  final def labels()
  : Array[Iterator[(Int, Real)]] = labels(Real.pointFive)

  final def labels(threshold: Real)
  : Array[Iterator[(Int, Real)]] = valuesMatrixEx match {
    case values: DenseMatrix[Real] =>
      values(::, breeze.linalg.*).map(
        _.iterator.filter(_._2 >= threshold)
      ).t.data
    case values: CSCMatrix[Real] =>
      val result = Array.fill(values.cols)(immutable.Vector.newBuilder[(Int, Real)])
      val iter = values.iterator
      while (iter.hasNext) {
        val ((index, sample), value) = iter.next()
        if (value >= threshold) {
          result(sample) += Tuple2(index, value)
        }
      }
      result.map(_.result().iterator)
  }

  final def topKLabels(k: Int)
  : Array[Iterator[(Int, Real)]] = valuesMatrixEx match {
    case values: DenseMatrix[Real] =>
      values(::, breeze.linalg.*).map(
        sampleValues => {
          val topK = new TopK[(Int, Real)](k)(Ordering.by(_._2))
          val iter = sampleValues.iterator
          while (iter.hasNext) {
            topK += iter.next()
          }
          topK.iterator
        }
      ).t.data
    case values: CSCMatrix[Real] =>
      val topK = Array.fill(values.cols)(
        new TopK[(Int, Real)](k)(Ordering.by(_._2))
      )
      val iter = values.iterator
      while (iter.hasNext) {
        val ((index, sampleNo), value) = iter.next()
        topK(sampleNo) += Tuple2(index, value)
      }
      ArrayEx.map(topK)(_.iterator)
  }

  def toValueTensor
  : ValueTensor

  def asOrToValueTensor
  : ValueTensor

  def toRealArrayTensor
  : RealArrayTensor

  def asOrToRealArrayTensor
  : RealArrayTensor

}

object Tensor {

  final def derive(part0: Tensor, parts: Tensor*)
  : Tensor = part0.concat(parts.toArray)

}

abstract class TensorEx[TThis <: Tensor]
  extends Tensor
    with CopyableEx[TThis] {

  def repr
  : TThis

  final override def createSibling()
  : TThis = createSibling(layout)

  override def createSibling(newLayout: TensorLayout)
  : TThis

  final override def createSiblingAndClear()
  : TThis = createSiblingAndClear(layout)

  override def createSiblingAndClear(newLayout: TensorLayout)
  : TThis

  override def copy
  : TThis


  // ---------------------------------------------------------------------------
  //    Basic operations.
  // ---------------------------------------------------------------------------
  override def reshape(newSize: Size): TThis

  override def apply(index: Int): TThis

  override def apply(indices: Range): TThis

  override def concat(other: Tensor): TThis

  final override def concat(other0: Tensor, others: Tensor*)
  : TThis = concat(SeqEx.concat(other0, others))

  override def concat[T <: Tensor](others: Array[T]): TThis

  final override def +(value: Real)
  : TThis = {
    val result = copy
    result += value
    result
  }

  final override def +(other: Tensor)
  : TThis = {
    val result = copy
    result += other
    result
  }

  final override def unary_-()
  : TThis = {
    val result = copy
    result.subtractR(Real.zero)
    result
  }

  final override def -(other: Tensor)
  : TThis = {
    val result = copy
    result -= other
    result
  }

  final override def *(value: Real)
  : TThis = {
    val result = copy
    result *= value
    result
  }

  final override def :*(other: Tensor)
  : TThis = {
    val result = copy
    result :*= other
    result
  }

  final override def :/(other: Tensor)
  : TThis = {
    val result = copy
    result :/= other
    result
  }

  final override def reciprocal()
  : TThis = {
    val result = copy
    result.divideR(Real.one)
    result
  }


  // ---------------------------------------------------------------------------
  //    Special concat and slice.
  // ---------------------------------------------------------------------------
  override def ++(other: Tensor): TThis

  override def :++(other: Tensor): TThis

  final override def slice(tuple0: Int, noTuples: Int)
  : TThis = slice(tuple0, layout.size.withNoTuples(noTuples))

  final override def slice(tuple0: Int, size: Size)
  : TThis = {
    val result = createSibling(layout.derive(size))
    slice(tuple0, result)
    result
  }

  final override def sliceChannels(channel0: Int, noChannels: Int)
  : TThis = sliceChannels(channel0, layout.size.withNoChannels(noChannels))

  final override def sliceChannels(channel0: Int, size: Size)
  : TThis = {
    val result = createSibling(layout.derive(size))
    sliceChannels(channel0, result)
    result
  }

}
