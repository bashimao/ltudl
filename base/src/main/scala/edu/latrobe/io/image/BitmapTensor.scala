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

package edu.latrobe.io.image

import breeze.linalg.DenseMatrix
import edu.latrobe._
import edu.latrobe.sizes._
import org.json4s.JsonAST._
import scala.reflect._
import scala.util.hashing._

/**
 * This is a temporary representation of a sample between augmentations. It is
 * not meant to be used somewhere else.
 */
// TODO: This type merely exists because I got tired of wrangling with pattern matching/type erasure issues. If you know how to do this better without cluttering the augmenter classes, please let me know!
final class BitmapTensor(val bitmaps: Array[Bitmap])
  extends TensorEx[BitmapTensor]
    with IndependentTensor {
  require(!ArrayEx.contains(bitmaps, null))

  override def repr
  : BitmapTensor = this

  override def toString
  : String = s"BitmapTensor[${bitmaps.length}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), ArrayEx.hashCode(bitmaps))

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: BitmapTensor =>
      ArrayEx.compare(bitmaps, other.bitmaps)
    case _ =>
      false
  })

  override def createSibling(layout: TensorLayout)
  : BitmapTensor = throw new UnsupportedOperationException

  override def createSiblingAndClear(layout: TensorLayout)
  : BitmapTensor = throw new UnsupportedOperationException

  override def copy
  : BitmapTensor = {
    val result = ArrayEx.map(
      bitmaps
    )(_.copy)
    BitmapTensor(result)
  }

  override protected def doClose()
  : Unit = {
    ArrayEx.foreach(
      bitmaps
    )(_.close())
    super.doClose()
  }


  // ---------------------------------------------------------------------------
  //    Data access related.
  // ---------------------------------------------------------------------------
  override def platform
  : JVM.type = JVM

  @transient
  override lazy val layout
  : IndependentTensorLayout = {
    if (bitmaps.length == 0) {
      IndependentTensorLayout.zero
    }
    else {
      val width      = bitmaps(0).width
      val height     = bitmaps(0).height
      val noChannels = bitmaps(0).noChannels
      assume(
        !ArrayEx.exists(
          bitmaps
        )(bmp => {
          bmp.width != width || bmp.height != height || bmp.noChannels != noChannels
        })
      )
      IndependentTensorLayout(Size2(width, height, noChannels), bitmaps.length)
    }
  }

  override def values
  : Array[Real] = {
    val size   = ArrayEx.foldLeft(0, bitmaps)(_ + _.size)
    val result = new Array[Real](size)
    get(result)
    result
  }

  override def valuesMatrix
  : DenseMatrix[Real] = {
    val l = layout
    new DenseMatrix(l.noValues, l.noSamples, values)
  }

  override def valuesMatrixEx
  : DenseMatrix[Real] = valuesMatrix

  override def get(valueNo: Int)
  : Real = {
    require(valueNo >= 0)
    var n = valueNo
    var i = 0
    while (i < bitmaps.length) {
      val bmp  = bitmaps(i)
      val size = bmp.size
      if (n < size) {
        val array = bmp.toRealArray
        return array(n)
      }
      n -= size
      i += 1
    }
    throw new IndexOutOfBoundsException
  }

  override def get(result: Array[Real], offset: Int, stride: Int)
  : Unit = {
    ArrayEx.foldLeft(offset, bitmaps)((off, bmp) => {
      val values = bmp.toRealArray
      ArrayEx.set(
        result, off, stride,
        values, 0,   1,
        values.length
      )
      off + values.length
    })
  }

  override def put(valueNo: Int, value: Real)
  : Unit = throw new UnsupportedOperationException

  override def put(array: Array[Real], offset: Int, stride: Int)
  : Unit = throw new UnsupportedOperationException

  override def clear()
  : Unit = {
    ArrayEx.foreach(
      bitmaps
    )(_.close())
  }

  override def fill(fn: () => Real, threadSafe: Boolean)
  : Unit = throw new UnsupportedOperationException


  // ---------------------------------------------------------------------------
  //    Basic operations.
  // ---------------------------------------------------------------------------
  override def reshape(newSize: Size)
  : BitmapTensor = throw new UnsupportedOperationException

  override def apply(index: Int)
  : BitmapTensor = {
    val slice = bitmaps(index)
    BitmapTensor.derive(slice.copy)
  }

  override def apply(indices: Range)
  : BitmapTensor = {
    val slice = ArrayEx.slice(bitmaps, indices)
    BitmapTensor(ArrayEx.deepCopy(slice))
  }

  override def splitSamples
  : Array[Tensor] = {
    ArrayEx.map(
      bitmaps
    )(bmp => BitmapTensor.derive(bmp.copy))
  }

  override def concat(other: Tensor)
  : BitmapTensor = other match {
    case other: BitmapTensor =>
      val result = ArrayEx.concat(
        ArrayEx.deepCopy(bitmaps),
        ArrayEx.deepCopy(other.bitmaps)
      )
      BitmapTensor(result)
    case _ =>
      throw new MatchError(other)
  }

  override def concat[T <: Tensor](others: Array[T])
  : BitmapTensor = {
    def getBitmaps(tensor: Tensor)
    : Array[Bitmap] = tensor match {
      case tensor: BitmapTensor =>
        ArrayEx.deepCopy(tensor.bitmaps)
      case _ =>
        throw new MatchError(tensor)
    }
    val result = ArrayEx.concat(
      ArrayEx.deepCopy(bitmaps),
      ArrayEx.map(
        others
      )(getBitmaps)
    )
    BitmapTensor(result)
  }

  override protected def doSet(value: Real)
  : Unit = throw new UnsupportedOperationException

  override protected def doSet(other: Tensor)
  : Unit = other match {
    case other: BitmapTensor =>
      ArrayEx.transform(
        bitmaps,
        other.bitmaps
      )((dst, src) => {
        dst.close()
        src.copy
      })
    case _ =>
      throw new MatchError(other)
  }

  override protected def doSet(other: Tensor, beta: Real)
  : Unit = throw new UnsupportedOperationException

  override protected def doSet(other0: Tensor, other1: Tensor)
  : Unit = throw new UnsupportedOperationException

  override protected def doSet(other0: Tensor, other1: Tensor, beta: Real)
  : Unit = throw new UnsupportedOperationException

  override protected def doAdd(value: Real)
  : Unit = throw new UnsupportedOperationException

  override protected def doAdd(other: Tensor)
  : Unit = throw new UnsupportedOperationException

  override protected def doAdd(alpha: Real,
                               other: Tensor)
  : Unit = throw new UnsupportedOperationException

  override protected def doAdd(other: Tensor, beta: Real)
  : Unit = throw new UnsupportedOperationException

  override protected def doAdd(alpha: Real,
                               other: Tensor, beta: Real)
  : Unit = throw new UnsupportedOperationException

  override protected def doAdd(other0: Tensor, other1: Tensor)
  : Unit = throw new UnsupportedOperationException

  override protected def doAdd(alpha:  Real,
                               other0: Tensor, other1: Tensor)
  : Unit = throw new UnsupportedOperationException

  override protected def doAdd(other0: Tensor, other1: Tensor, beta: Real)
  : Unit = throw new UnsupportedOperationException

  override protected def doAdd(alpha:  Real,
                               other0: Tensor, other1: Tensor, beta: Real)
  : Unit = throw new UnsupportedOperationException

  override protected def doSubtract(other: Tensor)
  : Unit = throw new UnsupportedOperationException

  override def subtractR(value: Real)
  : Unit = throw new UnsupportedOperationException

  override protected def doMultiply(value: Real)
  : Unit = throw new UnsupportedOperationException

  override protected def doMultiply(other: Tensor)
  : Unit = throw new UnsupportedOperationException

  override protected def doMultiply(other: Tensor, beta: Real)
  : Unit = throw new UnsupportedOperationException

  override protected def doDivide(other: Tensor)
  : Unit = throw new UnsupportedOperationException

  override protected def doDivide(epsilon0: Real,
                                  other:   Tensor)
  : Unit = throw new UnsupportedOperationException

  override protected def doDivide(other: Tensor, epsilon1: Real)
  : Unit = throw new NotImplementedError

  override protected def doDivide(epsilon0: Real,
                                  other:    Tensor, epsilon1: Real)
  : Unit = throw new NotImplementedError

  override protected def doDivideR(value: Real)
  : Unit = throw new UnsupportedOperationException

  override protected def doDot(other: Tensor)
  : Real = ArrayEx.dot(
    values,
    other.values
  )

  override protected def doLerp(other: Tensor, t: Real)
  : Unit = throw new UnsupportedOperationException

  override protected def doLerp(other0: Tensor, other1: Tensor, t: Real)
  : Unit = throw new UnsupportedOperationException


  // ---------------------------------------------------------------------------
  //    Fancy operations.
  // ---------------------------------------------------------------------------
  override def abs()
  : Unit = throw new UnsupportedOperationException

  override def approximateMean(rng:          PseudoRNG,
                               noSamplesMax: Int)
  : Mean = Mean.derive(values, rng, noSamplesMax)

  override def approximateMeanAndVariance(rng:          PseudoRNG,
                                          noSamplesMax: Int)
  : MeanAndVariance = MeanAndVariance.derive(values, rng, noSamplesMax)

  override def l1Norm(epsilon: Double)
  : Real = ArrayEx.l1Norm(values, epsilon)

  override def l2Norm(epsilon: Double)
  : Real = ArrayEx.l2Norm(values, epsilon)

  override def l2NormSq
  : Real = ArrayEx.l2NormSq(values)

  override def max()
  : Real = ArrayEx.max(values)

  override def max(other: Tensor)
  : Unit = throw new NotImplementedError

  override def maxAbs()
  : Real = throw new UnsupportedOperationException

  override protected def doMaxByAbs(other: Tensor)
  : Unit = throw new UnsupportedOperationException

  override def mean
  : Real = ArrayEx.mean(values)

  override def min()
  : Real = ArrayEx.min(values)

  override def min(other: Tensor)
  : Unit = throw new UnsupportedOperationException

  override def sign()
  : Unit = throw new UnsupportedOperationException

  override def sqr()
  : Unit = throw new UnsupportedOperationException

  override def sqrt()
  : Unit = throw new UnsupportedOperationException

  override def stdDev(epsilon: Double)
  : Real = ArrayEx.sampleStdDev(values, epsilon)

  override def sum
  : Real = ArrayEx.sum(values)

  def foreachBitmap(fn: Bitmap => Unit)
  : Unit = {
    if (bitmaps.length > 1) {
      ArrayEx.foreachParallel(
        bitmaps
      )(fn)
    }
    else {
      ArrayEx.foreach(
        bitmaps
      )(fn)
    }
  }

  def mapBitmaps[T](fn: Bitmap => T)
                   (implicit tagT: ClassTag[T])
  : Array[T] = {
    if (bitmaps.length > 1) {
      ArrayEx.mapParallel(
        bitmaps
      )(fn)
    }
    else {
      ArrayEx.map(
        bitmaps
      )(fn)
    }
  }

  override def ++(other: Tensor)
  : BitmapTensor = throw new NotImplementedError

  override def :++(other: Tensor)
  : BitmapTensor = throw new NotImplementedError

  override protected def doSlice(tuple0: Int,
                                 result: Tensor)
  : Unit = throw new NotImplementedError

  override protected def doSliceChannels(channel0: Int,
                                         result:   Tensor)
  : Unit = throw new NotImplementedError


  // ---------------------------------------------------------------------------
  //    Conversion & extraction methods.
  // ---------------------------------------------------------------------------
  override protected def doToJson()
  : List[JField] = List(
    Json.field("bitmaps", Json(bitmaps))
  )

  override def toValueTensor
  : ValueTensor = RealArrayTensor(layout, values)

  override def asOrToValueTensor
  : ValueTensor = RealArrayTensor(layout, values)

  override def toRealArrayTensor
  : RealArrayTensor = RealArrayTensor(layout, values)

  override def asOrToRealArrayTensor
  : RealArrayTensor = RealArrayTensor(layout, values)

}

object BitmapTensor {

  final def apply(bitmaps: Array[Bitmap])
  : BitmapTensor = new BitmapTensor(bitmaps)

  final def derive(bitmap0: Bitmap)
  : BitmapTensor = apply(Array(bitmap0))

  final def derive(bitmap0: Bitmap, bitmaps: Bitmap*)
  : BitmapTensor = apply(SeqEx.concat(bitmap0, bitmaps))

}
