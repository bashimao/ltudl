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

final class ConvertToReal(override val builder:        ConvertToRealBuilder,
                          override val inputHints:     BuildHints,
                          override val seed:           InstanceSeed,
                          override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends BitmapLayer[ConvertToRealBuilder] {

  override protected def doPredict(input: BitmapTensor)
  : RealArrayTensor = {
    val layout = input.layout
    val values = ArrayEx.concat(
      input.mapBitmaps(_.toRealArray)
    )
    RealArrayTensor(layout, values)
  }

}

final class ConvertToRealBuilder
  extends BitmapLayerBuilder[ConvertToRealBuilder] {

  override def repr
  : ConvertToRealBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ConvertToRealBuilder]

  override protected def doCopy()
  : ConvertToRealBuilder = ConvertToRealBuilder()

  override def outputSizeFor(sizeHint: Size)
  : Size = sizeHint

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : ConvertToReal = new ConvertToReal(this, hints, seed, weightsBuilder)

}

object ConvertToRealBuilder {

  final def apply()
  : ConvertToRealBuilder = new ConvertToRealBuilder

}