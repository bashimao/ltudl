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

/**
  * Simply zeros the entire tensor. Similar to "Set" but with less parameters.
  *
  * f(x_a) = 0
  *
  *       -1
  * f(x_a)   = impossible
  *
  *
  * d f(x_a)
  * -------- = 0
  *   x_a
  */
final class Zero(override val builder:             ZeroBuilder,
                 override val inputHints:          BuildHints,
                 override val seed:                InstanceSeed,
                 override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends NonTrainableMapLayer[ZeroBuilder]
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
  : (Tensor, PredictContext) = {
    if (inPlaceAllowed) {
      input.clear()
      (input, EmptyContext)
    }
    else {
      val out = input.createSiblingAndClear()
      (out, EmptyContext)
    }
  }

  override protected def doPredictInv(output:  Tensor,
                                      context: PredictContext)
  : Tensor = throw new UnsupportedOperationException


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
  : Tensor = {
    error.clear()
    error
  }

}

final class ZeroBuilder
  extends NonTrainableMapLayerBuilder[ZeroBuilder] {

  override def repr
  : ZeroBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ZeroBuilder]

  override protected def doCopy()
  : ZeroBuilder = ZeroBuilder()


  // ---------------------------------------------------------------------------
  //    Weights and binding related.
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(hints: BuildHints)
  : Platform = hints.platform

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Zero = new Zero(this, hints, seed, weightsBuilder)

}

object ZeroBuilder {

  final def apply()
  : ZeroBuilder = new ZeroBuilder

}
