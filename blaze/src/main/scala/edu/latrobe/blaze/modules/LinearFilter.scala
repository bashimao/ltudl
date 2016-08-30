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

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules.jvm._

abstract class LinearFilter
  extends LinearFilterLike[LinearFilterBuilder] {

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

  final override def extractWeightsFor(neuronNo: Int)
  : Array[Real] = filter.valuesMatrix(::, neuronNo).toArray

}

final class LinearFilterBuilder
  extends LinearFilterLikeBuilder[LinearFilterBuilder] {

  override def repr: LinearFilterBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LinearFilterBuilder]

  override protected def doCopy()
  : LinearFilterBuilder = LinearFilterBuilder()


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(hints: BuildHints)
  : Platform = LinearFilterBuilder.outputPlatformFor(this, hints)

  // Lookup variant and create object.
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = LinearFilterBuilder.lookupAndBuild(
    this, hints, seed, weightsBuilder
  )


  /**
   * @param noOutputs The reconstruction size of the decoder. Should be
   *                  equivalent to the inputSizeHint given to this instance.
   */
  def deriveCompatibleDecoder(noOutputs: Int)
  : LinearFilterDecoderBuilder = {
    val res = LinearFilterDecoderBuilder(noOutputs)
    res.setFilterReference(filterReference)
    res
  }

}

object LinearFilterBuilder
  extends ModuleVariantTable[LinearFilterBuilder] {

  register(2, LinearFilter_JVM_Breeze_Description)
  register(4, LinearFilter_JVM_BLAS_Description)

  final def apply()
  : LinearFilterBuilder = new LinearFilterBuilder

  final def apply(noOutputs: Int)
  : LinearFilterBuilder = apply().setNoOutputs(noOutputs)

}
