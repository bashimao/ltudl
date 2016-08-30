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

final class CropRandomPatch(override val builder:        CropRandomPatchBuilder,
                            override val inputHints:     BuildHints,
                            override val seed:           InstanceSeed,
                            override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends CropPatchLike[CropRandomPatchBuilder] {

  override protected def doPredict(input: BitmapTensor)
  : BitmapTensor = {
    val out = input.mapBitmaps(input => {
      val spaceX = input.width  - dims._1
      val spaceY = input.height - dims._2
      require(spaceX >= 0 && spaceY >= 0)

      val x = if (spaceX > 0) rng.nextInt(spaceX) else 0
      val y = if (spaceY > 0) rng.nextInt(spaceY) else 0

      // Shares data array with source bitmap.
      input.crop(x, y, dims._1, dims._2)
    })
    BitmapTensor(out)
  }

}

final class CropRandomPatchBuilder
  extends CropPatchLikeBuilder[CropRandomPatchBuilder] {

  override def repr
  : CropRandomPatchBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[CropRandomPatchBuilder]

  override protected def doCopy()
  : CropRandomPatchBuilder = CropRandomPatchBuilder()

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : CropRandomPatch = new CropRandomPatch(this, hints, seed, weightsBuilder)

}

object CropRandomPatchBuilder {

  final def apply(): CropRandomPatchBuilder = new CropRandomPatchBuilder

  final def apply(width: Int, height: Int)
  : CropRandomPatchBuilder = apply((width, height))

  final def apply(dims: (Int, Int))
  : CropRandomPatchBuilder = apply().setDims(dims)

}
