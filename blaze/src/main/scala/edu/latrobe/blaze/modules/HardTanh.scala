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
 * Hard approximation to Tanh function:
 *
 * Activation: hardtanh(x) = min(max(x, -1), 1)
 *
 *           d hardtanh   { x <= -1 => 0
 * Gradient: ---------- = { x >= 1  => 0
 *               d x      { else    => 1
 */
abstract class HardTanh
  extends NonTrainableMapLayer[HardTanhBuilder]
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

  protected def doPredict(inPlaceAllowed: Boolean, input: Tensor): Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = throw new UnsupportedOperationException


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.RequiresEither

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.RequiresEither

  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = {
    if (input != null) {
      doDeriveInputError(input, error)
    }
    else {
      doDeriveInputError(output, error)
    }
  }

  protected def doDeriveInputError(inputOrOutput: Tensor, error: Tensor): Tensor

}

final class HardTanhBuilder
  extends NonTrainableMapLayerBuilder[HardTanhBuilder] {

  override def repr
  : HardTanhBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[HardTanhBuilder]

  override protected def doCopy()
  : HardTanhBuilder = HardTanhBuilder()

  override def outputPlatformFor(hints: BuildHints)
  : Platform = HardTanhBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = HardTanhBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object HardTanhBuilder
  extends ModuleVariantTable[HardTanhBuilder] {

  register(2, HardTanh_JVM_Baseline_Description)

  final def apply(): HardTanhBuilder = new HardTanhBuilder

}
