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

package edu.latrobe.blaze.modules.bitmap

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.sizes._
import edu.latrobe.io.image._
import scala.util.hashing._

abstract class ResampleLike[TBuilder <: ResampleLikeBuilder[_]]
  extends BitmapLayer[TBuilder] {

  final val format = builder.format

}

abstract class ResampleLikeBuilder[TThis <: ResampleLikeBuilder[_]]
  extends BitmapLayerBuilder[TThis] {

  final private var _format
  : BitmapFormat = BitmapFormat.BGR

  final def format
  : BitmapFormat = _format

  final def format_=(value: BitmapFormat)
  : Unit = {
    require(value != null)
    _format = value
  }

  final def setFormat(value: BitmapFormat)
  : TThis = {
    format_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = _format :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _format.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ResampleLikeBuilder[_] =>
      _format == other._format
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ResampleLikeBuilder[_] =>
        other._format = _format
      case _ =>
    }
  }

  def outputDimsFor(inputDims: (Int, Int)): (Int, Int)

  final def outputSizeFor(sizeHint: Size)
  : Size2 = sizeHint match {
    case inputSize: Size2 =>
      Size2(outputDimsFor(inputSize.dims), _format.noChannels)
    case _ =>
      //Size2(sizeHint.noTuples, 1, _format.noChannels)
      throw new MatchError(sizeHint)
  }

}

/**
  * Unlike scale which just changes the size of the bitmap, this will also
  * homogenize the pixel format.
  */
final class Resample(override val builder:        ResampleBuilder,
                     override val inputHints:     BuildHints,
                     override val seed:           InstanceSeed,
                     override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends ResampleLike[ResampleBuilder] {

  val dims = builder.dims

  override protected def doPredict(input: BitmapTensor)
  : BitmapTensor = {
    val out = input.mapBitmaps(
      _.resample(dims, format)
    )
    BitmapTensor(out)
  }

}

final class ResampleBuilder
  extends ResampleLikeBuilder[ResampleBuilder] {

  override def repr
  : ResampleBuilder = this

  private var _dims
  : (Int, Int) = (128, 128)

  def dims
  : (Int, Int)  = _dims

  def dims_=(value: (Int, Int))
  : Unit = {
    require(value != null)
    _dims = value
  }

  def setDims(value: (Int, Int))
  : ResampleBuilder = {
    dims_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _dims :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _dims.hashCode())

  override def canEqual(that: Any): Boolean = that.isInstanceOf[ResampleBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ResampleBuilder =>
      _dims == other._dims
    case _ =>
      false
  })

  override protected def doCopy()
  : ResampleBuilder = ResampleBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ResampleBuilder =>
        other._dims = _dims
      case _ =>
    }
  }

  override def outputDimsFor(inputDims: (Int, Int))
  : (Int, Int) = _dims

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Resample = new Resample(this, hints, seed, weightsBuilder)

}

object ResampleBuilder {

  final def apply()
  : ResampleBuilder = new ResampleBuilder

  final def apply(dims: (Int, Int))
  : ResampleBuilder = apply().setDims(dims)

  final def apply(dims: (Int, Int), format: BitmapFormat)
  : ResampleBuilder = apply(dims).setFormat(format)

}

final class ResampleEx(override val builder:        ResampleExBuilder,
                       override val inputHints:     BuildHints,
                       override val seed:           InstanceSeed,
                       override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends ResampleLike[ResampleExBuilder] {

  val dimsFn = builder.dimsFn

  override protected def doPredict(input: BitmapTensor)
  : BitmapTensor = {
    val out = input.mapBitmaps(input => {
      input.resample(dimsFn(input.width, input.height), format)
    })
    BitmapTensor(out)
  }

}

final class ResampleExBuilder
  extends ResampleLikeBuilder[ResampleExBuilder] {

  override def repr
  : ResampleExBuilder = this

  private var _dimsFn
  : (Int, Int) => (Int, Int) = _

  def dimsFn
  : (Int, Int) => (Int, Int) = _dimsFn

  def dimsFn_=(value: (Int, Int) => (Int, Int))
  : Unit = {
    require(value != null)
    _dimsFn = value
  }

  def setDimsFn(value: (Int, Int) => (Int, Int))
  : ResampleExBuilder = {
    dimsFn_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _dimsFn :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _dimsFn.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ResampleExBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ResampleExBuilder =>
      _dimsFn == other._dimsFn
    case _ =>
      false
  })

  override protected def doCopy()
  : ResampleExBuilder = ResampleExBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ResampleExBuilder =>
        other._dimsFn = _dimsFn
      case _ =>
    }
  }

  override def outputDimsFor(inputDims: (Int, Int))
  : (Int, Int) = _dimsFn(inputDims._1, inputDims._2)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : ResampleEx = new ResampleEx(this, hints, seed, weightsBuilder)

}

object ResampleExBuilder {

  final def apply()
  : ResampleExBuilder = new ResampleExBuilder

  final def apply(dimsFn: (Int, Int) => (Int, Int))
  : ResampleExBuilder = apply().setDimsFn(dimsFn)

  final def apply(dimsFn: (Int, Int) => (Int, Int), format: BitmapFormat)
  : ResampleExBuilder = apply(dimsFn).setFormat(format)

  final def fixShortEdge(edgeLength: Int, format: BitmapFormat)
  : ResampleExBuilder = {
    require(edgeLength > 0)
    apply(
      (width, height) => {
        if (width < height) {
          (edgeLength, height * edgeLength / width)
        }
        else {
          (width * edgeLength / height, edgeLength)
        }
      },
      format
    )
  }

}
