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
import edu.latrobe.blaze.TensorDependency._

/**
  * Only active during construction.
  *
  * This is injected by generic modules to make the next layer prefer a
  * specific platform.
  */
final class ChangePlatformPreference(override val builder:        ChangePlatformPreferenceBuilder,
                                     override val inputHints:     BuildHints,
                                     override val seed:           InstanceSeed,
                                     override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Layer[ChangePlatformPreferenceBuilder]
    with NonTrainableLayer[ChangePlatformPreferenceBuilder]
    with NonPenalizing {

  override val outputHints
  : BuildHints = builder.outputHintsFor(inputHints)


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(mode:           Mode,
                                   inPlaceAllowed: Boolean,
                                   input:          Tensor,
                                   reference:      Tensor)
  : (Tensor, PredictContext) = (input, EmptyContext)

  override protected def doPredictInv(output: Tensor, context: PredictContext)
  : Tensor = output


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  override protected def doDeriveInputError(input:     Tensor,
                                            reference: Tensor,
                                            output:    Tensor,
                                            context:   PredictContext,
                                            error:     Tensor)
  : Tensor = error

}

final class ChangePlatformPreferenceBuilder
  extends LayerBuilder[ChangePlatformPreferenceBuilder]
    with NonTrainableLayerBuilder[ChangePlatformPreferenceBuilder] {

  override def repr
  : ChangePlatformPreferenceBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ChangePlatformPreferenceBuilder]

  override protected def doCopy()
  : ChangePlatformPreferenceBuilder = ChangePlatformPreferenceBuilder()


  // ---------------------------------------------------------------------------
  //    Weights and binding related.
  // ---------------------------------------------------------------------------
  override def weightLayoutFor(hints:   BuildHints,
                               builder: TensorLayoutBufferBuilder)
  : BuildHints = outputHintsFor(hints)

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = {
    val p = preferredPlatform.getOrElse(hints.platform)
    hints.withPreferenceFor(p)
  }

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : ChangePlatformPreference = new ChangePlatformPreference(
    this, hints, seed, weightsBuilder
  )

}

object ChangePlatformPreferenceBuilder {

  final def apply()
  : ChangePlatformPreferenceBuilder = new ChangePlatformPreferenceBuilder

  final def apply(preferredPlatform: Platform)
  : ChangePlatformPreferenceBuilder = apply().setPreferredPlatform(preferredPlatform)

}