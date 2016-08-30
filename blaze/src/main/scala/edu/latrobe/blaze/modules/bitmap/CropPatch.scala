/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2014 Matthias Langer (t3l@threelights.de)
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
 */

package edu.latrobe.blaze.modules.bitmap

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.sizes._
import edu.latrobe.io.image._
import scala.util.hashing._

abstract class CropPatchLike[TBuilder <: CropPatchLikeBuilder[_]]
  extends BitmapLayer[TBuilder] {

  final val dims = builder.dims

}

abstract class CropPatchLikeBuilder[TThis <: CropPatchLikeBuilder[_]]
  extends BitmapLayerBuilder[TThis] {

  final private var _dims
  : (Int, Int) = (128, 128)

  final def dims
  : (Int, Int) = _dims

  final def dims_=(value: (Int, Int))
  : Unit = {
    require(value._1 > 0)
    require(value._2 > 0)
    _dims = value
  }

  final def setDims(value: (Int, Int))
  : TThis = {
    dims_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = _dims :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _dims.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: CropPatchLikeBuilder[_] =>
      _dims == other._dims
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: CropPatchLikeBuilder[_] =>
        other._dims = _dims
      case _ =>
    }
  }

  def outputSizeFor(sizeHint: Size)
  : Size2 = Size2(dims, sizeHint.noChannels)

}

final class CropPatch(override val builder:        CropPatchBuilder,
                      override val inputHints:     BuildHints,
                      override val seed:           InstanceSeed,
                      override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends CropPatchLike[CropPatchBuilder] {

  val offset = builder.offset

  override protected def doPredict(input: BitmapTensor)
  : BitmapTensor = {
    val out = input.mapBitmaps(
      _.crop(offset, dims)
    )
    BitmapTensor(out)
  }

}

final class CropPatchBuilder
  extends CropPatchLikeBuilder[CropPatchBuilder] {

  override def repr
  : CropPatchBuilder = this

  private var _offset
  : (Int, Int) = (0, 0)

  def offset
  : (Int, Int) = _offset

  def offset_=(value: (Int, Int))
  : Unit = {
    require(value != null)
    _offset = value
  }

  def setOffset(value: (Int, Int))
  : CropPatchBuilder = {
    offset_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _offset :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _offset.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[CropPatchBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: CropPatchBuilder =>
      _offset == other._offset
    case _ =>
      false
  })

  override protected def doCopy()
  : CropPatchBuilder = CropPatchBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: CropPatchBuilder =>
        other._offset = _offset
      case _ =>
    }
  }

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : CropPatch = new CropPatch(this, hints, seed, weightsBuilder)

}

object CropPatchBuilder {

  final def apply()
  : CropPatchBuilder = new CropPatchBuilder

  final def apply(width: Int, height: Int)
  : CropPatchBuilder = apply((width, height))

  final def apply(dims: (Int, Int))
  : CropPatchBuilder = apply().setDims(dims)

  final def apply(dims: (Int, Int), offset: (Int, Int))
  : CropPatchBuilder = apply(dims).setOffset(offset)

}