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
import edu.latrobe.io.FileTensor
import edu.latrobe.sizes._
import edu.latrobe.io.image._

final class DecodeBitmaps(override val builder:        DecodeBitmapsBuilder,
                          override val inputHints:     BuildHints,
                          override val seed:           InstanceSeed,
                          override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends RawLayer[DecodeBitmapsBuilder]
    with NonTrainableLayer[DecodeBitmapsBuilder]
    with NonPenalizing {

  override protected def doPredict(input: RawTensor)
  : BitmapTensor = {
    val result = input match {
      case input: FileTensor =>
        input.mapSampleHandles(Bitmap.decode)
      case _ =>
        input.mapSampleBytes(Bitmap.decode)
    }
    BitmapTensor(result)
  }

}

final class DecodeBitmapsBuilder
  extends RawLayerBuilder[DecodeBitmapsBuilder] {

  override def repr
  : DecodeBitmapsBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[DecodeBitmapsBuilder]

  override protected def doCopy()
  : DecodeBitmapsBuilder = DecodeBitmapsBuilder()

  // ---------------------------------------------------------------------------
  //   Weights / Building related.
  // ---------------------------------------------------------------------------
  override def weightLayoutFor(hints:   BuildHints,
                               builder: TensorLayoutBufferBuilder)
  : BuildHints = outputHintsFor(hints)

  def outputSizeFor(sizeHint: Size)
  : Size2 = sizeHint match {
    case sizeHint: Size2 =>
      sizeHint.withNoChannels(3)
    case _ =>
      Size2(sizeHint.noTuples, 1, 3)
  }

  def outputLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = layoutHint.derive(outputSizeFor(layoutHint.size))

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = {
    val layout = outputLayoutFor(outputLayoutFor(hints.layout))
    hints.derive(JVM, layout)
  }

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : DecodeBitmaps = new DecodeBitmaps(this, hints, seed, weightsBuilder)

}

object DecodeBitmapsBuilder {

  final def apply()
  : DecodeBitmapsBuilder = new DecodeBitmapsBuilder

}
