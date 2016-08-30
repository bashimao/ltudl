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

final class CropCenterSquare(override val builder:        CropCenterSquareBuilder,
                             override val inputHints:     BuildHints,
                             override val seed:           InstanceSeed,
                             override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends BitmapLayer[CropCenterSquareBuilder] {

  override protected def doPredict(input: BitmapTensor)
  : BitmapTensor = {
    val out = input.mapBitmaps(
      _.cropCenterSquare()
    )
    BitmapTensor(out)
  }

}

final class CropCenterSquareBuilder
  extends BitmapLayerBuilder[CropCenterSquareBuilder] {

  override def repr
  : CropCenterSquareBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[CropCenterSquareBuilder]

  override protected def doCopy()
  : CropCenterSquareBuilder = CropCenterSquareBuilder()

  override def outputSizeFor(sizeHint: Size): Size2 = sizeHint match {
    case sizeHint: Size2 =>
      val tmp = Math.min(sizeHint.dims._1, sizeHint.dims._2)
      Size2(tmp, tmp, sizeHint.noChannels)
    case _ =>
      Size2(sizeHint.noTuples, 1, sizeHint.noChannels)
  }

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : CropCenterSquare = new CropCenterSquare(this, hints, seed, weightsBuilder)

}

object CropCenterSquareBuilder {

  final def apply()
  : CropCenterSquareBuilder = new CropCenterSquareBuilder

}