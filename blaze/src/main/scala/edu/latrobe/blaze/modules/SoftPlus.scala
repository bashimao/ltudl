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
 * SoftPlus function
 * (differentiable approximation of rectified linear)
 *                             (      x )
 * Activation: softPlus(x) = ln( 1 + e  )
 *                             (        )
 *
 *                     -1     (  x     )
 * Inverse: softPlus(x)   = ln( e  - 1 )
 *                            (        )
 *
 *                             1
 * Gradient: softPlus(x)' = -------  = sigmoid(x)
 *                               -x
 *                          1 + e
 */
abstract class SoftPlus
  extends NonTrainableMapLayer[SoftPlusBuilder]
    with NonPenalizing {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredict(input)
    (out, EmptyContext)
  }

  protected def doPredict(input: Tensor): Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = doPredictInv(output)

  protected def doPredictInv(output: Tensor): Tensor


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.Required

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = doDeriveInputError(input, error)

  protected def doDeriveInputError(input: Tensor, error: Tensor): Tensor

}

final class SoftPlusBuilder
  extends NonTrainableMapLayerBuilder[SoftPlusBuilder] {

  override def repr
  : SoftPlusBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[SoftPlusBuilder]

  override protected def doCopy()
  : SoftPlusBuilder = SoftPlusBuilder()

  override def outputPlatformFor(hints: BuildHints)
  : Platform = SoftPlusBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = SoftPlusBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object SoftPlusBuilder
  extends ModuleVariantTable[SoftPlusBuilder] {

  register(2, SoftPlus_JVM_Baseline_Description)

  final def apply(): SoftPlusBuilder = new SoftPlusBuilder

}
