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

/**
 * This is a special dummy layer class. Each input neuron is connected
 * to directly to one output neuron. In other words: It does absolutely nothing!
 *
 * Prediction: x
 *
 * Inverse: x
 *
 * Gradient: 1
 */
final class Identity(override val builder:        IdentityBuilder,
                     override val inputHints:     BuildHints,
                     override val seed:           InstanceSeed,
                     override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends NonTrainableMapLayer[IdentityBuilder]
    with NonPenalizing {

  override lazy val outputPlatform
  : Platform = inputHints.platform


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(mode:           Mode,
                                   inPlaceAllowed: Boolean,
                                   input:          Tensor,
                                   reference:      Tensor)
  : (Tensor, PredictContext) = (input, EmptyContext)

  override protected def doPredictInv(output:  Tensor,
                                      context: PredictContext)
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

final class IdentityBuilder
  extends NonTrainableMapLayerBuilder[IdentityBuilder] {

  override def repr
  : IdentityBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[IdentityBuilder]

  override protected def doCopy()
  : SelectInputBuilder = IdentityBuilder()


  // ---------------------------------------------------------------------------
  //    Weights and binding related.
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(hints: BuildHints)
  : Platform = hints.platform

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Identity = new Identity(this, hints, seed, weightsBuilder)

}

object IdentityBuilder {

  final def apply()
  : IdentityBuilder = new IdentityBuilder

}
