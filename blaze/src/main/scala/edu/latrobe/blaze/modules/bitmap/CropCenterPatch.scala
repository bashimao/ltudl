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

final class CropCenterPatch(override val builder:        CropCenterPatchBuilder,
                            override val inputHints:     BuildHints,
                            override val seed:           InstanceSeed,
                            override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends CropPatchLike[CropCenterPatchBuilder] {

  override protected def doPredict(input: BitmapTensor)
  : BitmapTensor = {
    val out = input.mapBitmaps(
      _.cropCenter(dims._1, dims._2)
    )
    BitmapTensor(out)
  }

}

final class CropCenterPatchBuilder
  extends CropPatchLikeBuilder[CropCenterPatchBuilder] {

  override def repr
  : CropCenterPatchBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[CropCenterPatchBuilder]

  override protected def doCopy()
  : CropCenterPatchBuilder = CropCenterPatchBuilder()

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : CropCenterPatch = new CropCenterPatch(this, hints, seed, weightsBuilder)

}

object CropCenterPatchBuilder {

  final def apply()
  : CropCenterPatchBuilder = new CropCenterPatchBuilder

  final def apply(width: Int, height: Int)
  : CropCenterPatchBuilder = apply((width, height))

  final def apply(dims: (Int, Int))
  : CropCenterPatchBuilder = apply().setDims(dims)

}
