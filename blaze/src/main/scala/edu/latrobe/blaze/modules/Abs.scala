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

/**
 * Absolute function. (not differentiable)
 */
abstract class Abs
  extends NonTrainableMapLayer[AbsBuilder]
    with NonPenalizing {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = (doPredict(input), EmptyContext)

  protected def doPredict(input: Tensor): Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = throw new UnsupportedOperationException


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.Required

  override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  override protected def doDeriveInputError(input:     Tensor,
                                            reference: Tensor,
                                            output:    Tensor,
                                            context:   PredictContext,
                                            error:     Tensor)
  : Tensor = doDeriveInputError(input, error)

  protected def doDeriveInputError(input: Tensor,
                                   error: Tensor)
  : Tensor

}

final class AbsBuilder
  extends NonTrainableMapLayerBuilder[AbsBuilder] {

  override def repr
  : AbsBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[AbsBuilder]

  override protected def doCopy()
  : AbsBuilder = AbsBuilder()

  override def outputPlatformFor(hints: BuildHints)
  : Platform = AbsBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = AbsBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object AbsBuilder
  extends ModuleVariantTable[AbsBuilder] {

  register(2, Abs_JVM_Baseline_Description)

  final def apply(): AbsBuilder = new AbsBuilder

}
