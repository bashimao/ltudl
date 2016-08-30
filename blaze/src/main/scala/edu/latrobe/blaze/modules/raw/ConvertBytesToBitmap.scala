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

package edu.latrobe.blaze.modules.raw

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.sizes._
import edu.latrobe.io.image._
import scala.util.hashing._

/**
 * The ugly sibling of "DecodeBitmap". This works with uncompressed raw image
 * data. Use this for MNIST!
 */
final class ConvertBytesToBitmap(override val builder:        ConvertBytesToBitmapBuilder,
                                 override val inputHints:     BuildHints,
                                 override val seed:           InstanceSeed,
                                 override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends RawLayer[ConvertBytesToBitmapBuilder]
    with NonTrainableLayer[ConvertBytesToBitmapBuilder]
    with NonPenalizing {

  val dims: (Int, Int) = builder.dims

  val format: BitmapFormat = builder.format

  override protected def doPredict(input: RawTensor)
  : BitmapTensor = {
    val result = input.mapSampleBytes(
      Bitmap.derive(dims._1, dims._2, format, _)
    )
    BitmapTensor(result)
  }

}

final class ConvertBytesToBitmapBuilder
  extends RawLayerBuilder[ConvertBytesToBitmapBuilder] {

  override def repr
  : ConvertBytesToBitmapBuilder = this

  private var _dims
  : (Int, Int) = (128, 128)

  def dims
  : (Int, Int) = _dims

  def dims_=(value: (Int, Int))
  : Unit = {
    require(value != null)
    _dims = value
  }

  def setDims(value: (Int, Int))
  : ConvertBytesToBitmapBuilder = {
    dims_=(value)
    this
  }

  private var _format
  : BitmapFormat = BitmapFormat.BGR

  def format
  : BitmapFormat = _format

  def format_=(value: BitmapFormat)
  : Unit = {
    require(value != null)
    _format = value
  }

  def setFormat(value: BitmapFormat)
  : ConvertBytesToBitmapBuilder = {
    format_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _dims :: _format :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ConvertBytesToBitmapBuilder]

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _dims.hashCode())
    tmp = MurmurHash3.mix(tmp, _format.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ConvertBytesToBitmapBuilder =>
      _dims   == other._dims &&
      _format == other._format
    case _ =>
      false
  })

  override protected def doCopy()
  : ConvertBytesToBitmapBuilder = ConvertBytesToBitmapBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ConvertBytesToBitmapBuilder =>
        other._dims   = _dims
        other._format = _format
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //   Weights / Building related.
  // ---------------------------------------------------------------------------
  override def weightLayoutFor(hints:   BuildHints,
                               builder: TensorLayoutBufferBuilder)
  : BuildHints = outputHintsFor(hints)

  def outputSize
  : Size2 = Size2(_dims, _format.noChannels)

  def outputLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = layoutHint.derive(outputSize)

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = hints.derive(JVM, outputLayoutFor(hints.layout))

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : ConvertBytesToBitmap = new ConvertBytesToBitmap(
    this, hints, seed, weightsBuilder
  )

}

object ConvertBytesToBitmapBuilder {

  final def apply()
  : ConvertBytesToBitmapBuilder = new ConvertBytesToBitmapBuilder

  final def apply(dims: (Int, Int), format: BitmapFormat)
  : ConvertBytesToBitmapBuilder = apply().setDims(dims).setFormat(format)

}