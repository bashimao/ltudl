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
 * Sigmoid function:
 *                           1
 * Activation: sigm(x) = ---------
 *                             -cx
 *                        1 + e
 *
 *                 -1      1    (  1      )
 * Inverse: sigm(x)   = - --- ln( --- - 1 )
 *                         c    (  x      )
 *
 *                             -cx
 *                          c e
 * Gradient: sigm(x)' = -------------- = sigm(x) (1 - sigm(x)) c
 *                                  2
 *                      (      -cx )
 *                      ( 1 + e    )
 */
// TODO: Add a variant that can make use of yeppp!
abstract class Sigmoid
  extends NonTrainableMapLayer[SigmoidBuilder]
    with NonPenalizing {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = doPredict(inPlaceAllowed, input)
    (out, EmptyContext)
  }

  protected def doPredict(inPlaceAllowed: Boolean, input: Tensor)
  : Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = doPredictInv(output)

  protected def doPredictInv(output: Tensor)
  : Tensor


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.Required

  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = doDeriveInputError(output, error)

  protected def doDeriveInputError(output: Tensor, error: Tensor)
  : Tensor

}

final class SigmoidBuilder
  extends NonTrainableMapLayerBuilder[SigmoidBuilder] {

  override def repr
  : SigmoidBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[SigmoidBuilder]

  override protected def doCopy()
  : SigmoidBuilder = SigmoidBuilder()

  override def outputPlatformFor(hints: BuildHints)
  : Platform = SigmoidBuilder.outputPlatformFor(this, hints)

  // Lookup variant and create object.
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = SigmoidBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object SigmoidBuilder
  extends ModuleVariantTable[SigmoidBuilder] {

  register(2, Sigmoid_JVM_ApacheCommons_Description)
  register(4, Sigmoid_JVM_Baseline_Description)

  final def apply()
  : SigmoidBuilder = new SigmoidBuilder

}
