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

import breeze.linalg.{DenseMatrix, Matrix}
import scala.reflect._

/**
  * A tensor that contains binary data.
  */
trait RawTensor
  extends Tensor
    with IndependentTensor {

  def sizeHint: Size

  override def createSibling()
  : RawTensor

  override def createSibling(newLayout: TensorLayout)
  : RawTensor

  override def createSiblingAndClear()
  : RawTensor

  override def createSiblingAndClear(newLayout: TensorLayout)
  : RawTensor

  override def copy
  : RawTensor


  // ---------------------------------------------------------------------------
  //    Data access related.
  // ---------------------------------------------------------------------------
  final override def values
  : Array[Real] = throw new UnsupportedOperationException

  final override def valuesMatrix
  : DenseMatrix[Real] = throw new UnsupportedOperationException

  final override def valuesMatrixEx
  : Matrix[Real] with Serializable = throw new UnsupportedOperationException

  /**
    * A byte array containing the tensor data. Might be new array or an array
    * to an internal state.
    */
  def bytes
  : Array[Array[Byte]]

  final override def get(valueNo: Int)
  : Real = throw new UnsupportedOperationException

  final override def put(valueNo: Int, value: Real)
  : Unit = throw new UnsupportedOperationException

  final override def get(result: Array[Real], offset: Int, stride: Int)
  : Unit = throw new UnsupportedOperationException

  final override def put(array: Array[Real], offset: Int, stride: Int)
  : Unit = throw new UnsupportedOperationException

  final override def fill(fn: () => Real, threadSafe: Boolean)
  : Unit = throw new UnsupportedOperationException


  // ---------------------------------------------------------------------------
  //    Basic operations.
  // ---------------------------------------------------------------------------
  override def reshape(size: Size)
  : RawTensor

  override def apply(index: Int)
  : RawTensor

  override def apply(indices: Range)
  : RawTensor

  override def concat(other: Tensor)
  : RawTensor

  override def concat[T <: Tensor](others: Array[T])
  : RawTensor

  final override protected def doSet(value: Real)
  : Unit = throw new UnsupportedOperationException

  final override protected def doSet(other: Tensor, beta: Real)
  : Unit = throw new UnsupportedOperationException

  final override protected def doSet(other0: Tensor, other1: Tensor)
  : Unit = throw new UnsupportedOperationException

  final override protected def doSet(other0: Tensor, other1: Tensor, beta: Real)
  : Unit = throw new UnsupportedOperationException

  override def +(value: Real)
  : RawTensor

  override def +(other: Tensor)
  : RawTensor

  final override protected def doAdd(value: Real)
  : Unit = throw new UnsupportedOperationException

  override protected def doAdd(other: Tensor)
  : Unit = throw new UnsupportedOperationException

  final override protected def doAdd(alpha: Real,
                                     other: Tensor)
  : Unit = throw new UnsupportedOperationException

  final override protected def doAdd(other: Tensor, beta: Real)
  : Unit = throw new UnsupportedOperationException

  final override protected def doAdd(alpha: Real,
                                     other: Tensor, beta: Real)
  : Unit = throw new UnsupportedOperationException

  final override protected def doAdd(other0: Tensor, other1: Tensor)
  : Unit = throw new UnsupportedOperationException

  final override protected def doAdd(alpha:  Real,
                                     other0: Tensor, other1: Tensor)
  : Unit = throw new UnsupportedOperationException

  final override protected def doAdd(other0: Tensor, other1: Tensor, beta: Real)
  : Unit = throw new UnsupportedOperationException

  final override protected def doAdd(alpha:  Real,
                                     other0: Tensor, other1: Tensor, beta: Real)
  : Unit = throw new UnsupportedOperationException

  override def unary_-()
  : RawTensor

  override def -(other: Tensor)
  : RawTensor

  final override protected def doSubtract(other: Tensor)
  : Unit = throw new UnsupportedOperationException

  final override def subtractR(value: Real)
  : Unit = throw new UnsupportedOperationException

  override def *(value: Real)
  : RawTensor

  override def :*(other: Tensor)
  : RawTensor

  final override protected def doMultiply(value: Real)
  : Unit = throw new UnsupportedOperationException

  final override protected def doMultiply(other: Tensor)
  : Unit = throw new UnsupportedOperationException

  final override protected def doMultiply(other: Tensor, beta: Real)
  : Unit = throw new UnsupportedOperationException

  override def :/(other: Tensor)
  : RawTensor

  final override protected def doDivide(other: Tensor)
  : Unit = throw new UnsupportedOperationException

  final override protected def doDivide(epsilon0: Real,
                                        other:    Tensor)
  : Unit = throw new UnsupportedOperationException

  final override protected def doDivide(other:    Tensor,
                                        epsilon1: Real)
  : Unit = throw new UnsupportedOperationException

  final override protected def doDivide(epsilon0: Real,
                                        other:    Tensor, epsilon1: Real)
  : Unit = throw new UnsupportedOperationException

  final override protected def doDivideR(value: Real)
  : Unit = throw new UnsupportedOperationException

  override def reciprocal()
  : RawTensor

  final override protected def doDot(other: Tensor)
  : Real = throw new UnsupportedOperationException

  final override protected def doLerp(other: Tensor, t: Real)
  : Unit = throw new UnsupportedOperationException

  final override protected def doLerp(other0: Tensor, other1: Tensor, t: Real)
  : Unit = throw new UnsupportedOperationException


  // ---------------------------------------------------------------------------
  //    Fancy operations.
  // ---------------------------------------------------------------------------
  final override def abs()
  : Unit = throw new UnsupportedOperationException

  final override def approximateMean(rng:          PseudoRNG,
                                     noSamplesMax: Int)
  : Mean = throw new UnsupportedOperationException

  final override def approximateMeanAndVariance(rng:          PseudoRNG,
                                                noSamplesMax: Int)
  : MeanAndVariance = throw new UnsupportedOperationException

  final override def l1Norm(epsilon: Double)
  : Real = throw new UnsupportedOperationException

  final override def l2Norm(epsilon: Double)
  : Real = throw new UnsupportedOperationException

  final override def l2NormSq
  : Real = throw new UnsupportedOperationException

  final override def max()
  : Real = throw new UnsupportedOperationException

  final override def max(other: Tensor)
  : Unit = throw new UnsupportedOperationException

  final override def maxAbs()
  : Real = throw new UnsupportedOperationException

  final override def mean
  : Real = throw new UnsupportedOperationException

  final override def min()
  : Real = throw new UnsupportedOperationException

  final override def min(other: Tensor)
  : Unit = throw new UnsupportedOperationException

  final override protected def doMaxByAbs(other: Tensor)
  : Unit = throw new UnsupportedOperationException

  final override def sign()
  : Unit = throw new UnsupportedOperationException

  final override def sqr()
  : Unit = throw new UnsupportedOperationException

  final override def sqrt()
  : Unit = throw new UnsupportedOperationException

  final override def stdDev(epsilon: Double)
  : Real = throw new UnsupportedOperationException

  final override def sum
  : Real = throw new UnsupportedOperationException

  final def mapSampleBytes[T](fn: Array[Byte] => T)
                             (implicit tagT: ClassTag[T])
  : Array[T] = {
    if (layout.noSamples > 1) {
      ArrayEx.mapParallel(
        bytes
      )(fn)
    }
    else {
      ArrayEx.map(
        bytes
      )(fn)
    }
  }


  // ---------------------------------------------------------------------------
  //    Slicing.
  // ---------------------------------------------------------------------------
  override def ++(other: Tensor)
  : RawTensor

  override def :++(other: Tensor)
  : RawTensor

  final override protected def doSlice(tuple0: Int,
                                       result: Tensor)
  : Unit = throw new UnsupportedOperationException

  final override protected def doSliceChannels(channel0: Int,
                                               result:   Tensor)
  : Unit = throw new UnsupportedOperationException


  // ---------------------------------------------------------------------------
  //    Conversion & extraction methods.
  // ---------------------------------------------------------------------------
  final override def toValueTensor
  : RealArrayTensor = RealArrayTensor(layout, values)

  final override def asOrToValueTensor
  : RealArrayTensor = RealArrayTensor(layout, values)

  final override def toRealArrayTensor
  : RealArrayTensor = RealArrayTensor(layout, values)

  final override def asOrToRealArrayTensor
  : RealArrayTensor = RealArrayTensor(layout, values)

}
