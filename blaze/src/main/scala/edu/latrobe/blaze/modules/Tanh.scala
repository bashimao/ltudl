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
import edu.latrobe.blaze.TensorDependency._

/**
 * Tanh function:
 *
 * Activation: tanh(c * x)
 *
 *                         1    ( 1 + c * x )
 * Inverse: atanh(c * x) = - ln ( --------- )
 *                         2    ( 1 - c * x )
 *
 *           d tanh                      2       (               2 )
 * Gradient: ------ = c - c * tanh(c * x)  = c * (1 - tanh(c * x)  )
 *            d x                                (                 )
 */
abstract class Tanh
  extends NonTrainableMapLayer[TanhBuilder]
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

final class TanhBuilder
  extends NonTrainableMapLayerBuilder[TanhBuilder] {

  override def repr
  : TanhBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[TanhBuilder]

  override protected def doCopy()
  : TanhBuilder = TanhBuilder()


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(hints: BuildHints)
  : Platform = TanhBuilder.outputPlatformFor(this, hints)

  // Lookup variant and create object.
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = TanhBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object TanhBuilder
  extends ModuleVariantTable[TanhBuilder] {

  register(2, Tanh_JVM_ApacheCommons_Description)
  register(4, Tanh_JVM_Baseline_Description)

  final def apply()
  : TanhBuilder = new TanhBuilder

}
