/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2015 Matthias Langer (t3l@threelights.de)
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
import edu.latrobe.blaze.TensorDependency._

/**
 *           x
 * Predict: e
 *
 * Inverse: ln(x)
 *
 *            x
 * Gradient: e
 */
abstract class Exp
  extends NonTrainableMapLayer[ExpBuilder]
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

  protected def doPredictInv(output:Tensor)
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
  : Tensor = {
    error :*= output
    error
  }

}

final class ExpBuilder
  extends NonTrainableMapLayerBuilder[ExpBuilder] {

  override def repr
  : ExpBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ExpBuilder]

  override protected def doCopy()
  : ExpBuilder = ExpBuilder()

  override def outputPlatformFor(hints: BuildHints)
  : Platform = ExpBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = ExpBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object ExpBuilder extends ModuleVariantTable[ExpBuilder] {

  register(2, Exp_JVM_ApacheCommons_Description)
  register(4, Exp_JVM_Baseline_Description)

  final def apply()
  : ExpBuilder = new ExpBuilder

}
