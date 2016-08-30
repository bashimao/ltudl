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

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules.jvm._

abstract class LinearFilterDecoder
  extends LinearFilterLike[LinearFilterDecoderBuilder] {

  // ---------------------------------------------------------------------------
  //    Weights related.
  // ---------------------------------------------------------------------------
  final override def reset(initializer: Initializer)
  : Unit = {
    val inputFanSize  = inputSizeHint.noValues
    val outputFanSize = outputSize.noValues
    filterReference.foreach(
      initializer(this, _, filter, inputFanSize, outputFanSize)
    )
  }

  override def extractWeightsFor(neuronNo: Int)
  : Array[Real] = filter.valuesMatrix(neuronNo, ::).inner.toArray

}

final class LinearFilterDecoderBuilder
  extends LinearFilterLikeBuilder[LinearFilterDecoderBuilder] {

  override def repr
  : LinearFilterDecoderBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LinearFilterDecoderBuilder]

  override protected def doCopy()
  : LinearFilterDecoderBuilder = LinearFilterDecoderBuilder()


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(hints: BuildHints)
  : Platform = LinearFilterDecoderBuilder.outputPlatformFor(this, hints)

  // Lookup variant and create object.
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = LinearFilterDecoderBuilder.lookupAndBuild(
    this, hints, seed, weightsBuilder
  )

  def deriveCompatibleEncoder(noOutputs: Int)
  : LinearFilterBuilder = {
    val res = LinearFilterBuilder(noOutputs)
    res.setFilterReference(filterReference)
    res
  }

}

object LinearFilterDecoderBuilder
  extends ModuleVariantTable[LinearFilterDecoderBuilder] {

  register(2, LinearFilterDecoder_JVM_Breeze_Description)

  final def apply()
  : LinearFilterDecoderBuilder = new LinearFilterDecoderBuilder

  final def apply(noOutputs: Int)
  : LinearFilterDecoderBuilder = apply().setNoOutputs(noOutputs)

}
