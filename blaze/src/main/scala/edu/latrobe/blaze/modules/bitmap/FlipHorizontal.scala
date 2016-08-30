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
import edu.latrobe.io.image._

final class FlipHorizontal(override val builder:        FlipHorizontalBuilder,
                           override val inputHints:     BuildHints,
                           override val seed:           InstanceSeed,
                           override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends BitmapLayer[FlipHorizontalBuilder] {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input: BitmapTensor)
  : BitmapTensor = {
    val out = input.mapBitmaps(
      _.flipHorizontal()
    )
    BitmapTensor(out)
  }

}

final class FlipHorizontalBuilder
  extends BitmapLayerBuilder[FlipHorizontalBuilder] {

  override def repr
  : FlipHorizontalBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[FlipHorizontalBuilder]

  override protected def doCopy()
  : FlipHorizontalBuilder = FlipHorizontalBuilder()

  override def outputSizeFor(sizeHint: Size)
  : Size = sizeHint

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : FlipHorizontal = new FlipHorizontal(this, hints, seed, weightsBuilder)

}

object FlipHorizontalBuilder {

  final def apply()
  : FlipHorizontalBuilder = new FlipHorizontalBuilder

}
