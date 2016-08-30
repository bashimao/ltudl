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

final class Copy(override val builder:             CopyBuilder,
                 override val inputHints:          BuildHints,
                 override val seed:                InstanceSeed,
                 override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends NonTrainableMapLayer[CopyBuilder]
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
  : (Tensor, PredictContext) = (input.copy, EmptyContext)

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

final class CopyBuilder
  extends NonTrainableMapLayerBuilder[CopyBuilder] {

  override def repr
  : CopyBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[CopyBuilder]

  override protected def doCopy()
  : CopyBuilder = CopyBuilder()


  // ---------------------------------------------------------------------------
  //    Weights and binding related.
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(hints: BuildHints)
  : Platform = hints.platform

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Copy = new Copy(this, hints, seed, weightsBuilder)

}

object CopyBuilder {

  final def apply()
  : CopyBuilder = new CopyBuilder

}
