/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2015 Matthias Langer (t3l@threelights.de)
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
import edu.latrobe.io.image._
import edu.latrobe.sizes._

final class ResampleAsRGB(override val builder:        ResampleAsRGBBuilder,
                          override val inputHints:     BuildHints,
                          override val seed:           InstanceSeed,
                          override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends BitmapLayer[ResampleAsRGBBuilder] {

  override protected def doPredict(input: BitmapTensor)
  : BitmapTensor = {
    val out = input.mapBitmaps(input => {
      input.format match {
        case BitmapFormat.BGR =>
          input.copy
        case _ =>
          input.resample(input.width, input.height, BitmapFormat.BGR)
      }
    })
    BitmapTensor(out)
  }

}

final class ResampleAsRGBBuilder
  extends BitmapLayerBuilder[ResampleAsRGBBuilder] {

  override def repr
  : ResampleAsRGBBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ResampleAsRGBBuilder]

  override protected def doCopy()
  : ResampleAsRGBBuilder = ResampleAsRGBBuilder()

  def outputSizeFor(sizeHint: Size)
  : Size2 = sizeHint match {
    case sizeHint: Size2 =>
      sizeHint.withNoChannels(3)
    case _ =>
      throw new MatchError(sizeHint)
  }

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : ResampleAsRGB = new ResampleAsRGB(this, hints, seed, weightsBuilder)

}

object ResampleAsRGBBuilder {

  final def apply(): ResampleAsRGBBuilder = new ResampleAsRGBBuilder

}