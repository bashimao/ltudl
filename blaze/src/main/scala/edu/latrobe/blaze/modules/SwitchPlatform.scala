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

/**
  * Allows switching deliberately to another platform. This is useful
  * when you want to make sure a layer gets its input in a certain tensor
  * format. (performance!)
  *
  * If CPU selected: Switches platform to CPU if not already CPU.
  * If CUDA selected: Switches platform to CUDA if not already on CUDA.
  */
abstract class SwitchPlatform
  extends NonTrainableMapLayer[SwitchPlatformBuilder]
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
    if (out eq input) {
      (out, EmptyContext)
    }
    else {
      (out, SwitchPlatformContext(input))
    }
  }

  protected def doPredict(input: Tensor)
  : Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = context match {
    case context: SwitchPlatformContext =>
      val inp = context.input.createSibling()
      inp := output
      inp
    case _ =>
      output
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = context match {
    case context: SwitchPlatformContext =>
      val newErr = context.input.createSibling()
      newErr := error
      newErr
    case _ =>
      error
  }

}

final class SwitchPlatformBuilder
  extends NonTrainableMapLayerBuilder[SwitchPlatformBuilder] {

  override def repr
  : SwitchPlatformBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[SwitchPlatformBuilder]

  override protected def doCopy()
  : SwitchPlatformBuilder = SwitchPlatformBuilder()

  override def outputPlatformFor(hints: BuildHints)
  : Platform = SwitchPlatformBuilder.outputPlatformFor(this, hints)

  override def build(hints:                BuildHints,
                     seed:                 InstanceSeed,
                     weightsBufferBuilder: ValueTensorBufferBuilder)
  : Module = SwitchPlatformBuilder.lookupAndBuild(
    this,
    hints,
    seed,
    weightsBufferBuilder
  )

}

object SwitchPlatformBuilder
  extends ModuleVariantTable[SwitchPlatformBuilder] {

  register(2, SwitchPlatform_JVM_Baseline_Description)

  final def apply()
  : SwitchPlatformBuilder = new SwitchPlatformBuilder

  final def apply(preferredPlatform: IndependentPlatform)
  : SwitchPlatformBuilder = apply().setPreferredPlatform(preferredPlatform)

}

final case class SwitchPlatformContext(input: Tensor)
  extends PredictContext {
  // We only keep this tensor for the allocate function.
}
