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

/**
 * Inserts a EmptyTensor on the forward pass, causing the underlying tensor
 * collector to deallocate tensors that are no longer required. Transparent
 * on the backward path.
 *
 * This is just a dummy layer that is inserted by forking layers to ensure
 * deallocation is conducted properly.
 */
final class BranchTerminator(override val builder:             BranchTerminatorBuilder,
                             override val inputHints:          BuildHints,
                             override val seed:                InstanceSeed,
                             override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Layer[BranchTerminatorBuilder]
    with NonTrainableLayer[BranchTerminatorBuilder]
    with NonPenalizing {

  override val outputHints
  : BuildHints = BuildHints.zero


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(mode:           Mode,
                                   inPlaceAllowed: Boolean,
                                   input:          Tensor,
                                   reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = RealArrayTensor.zeros(IndependentTensorLayout.zero)
    val ctx = BranchTerminatorContext(input)
    (out, ctx)
  }

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

  override protected def doDeriveInputError(_input:    Tensor,
                                            reference: Tensor,
                                            output:    Tensor,
                                            context:   PredictContext,
                                            error:     Tensor)
  : Tensor = context match {
    case BranchTerminatorContext(input) =>
      if (error == null) {
        input.createSiblingAndClear()
      }
      else {
        error
      }
    case _ =>
      throw new MatchError(context)
  }

}

final class BranchTerminatorBuilder
  extends LayerBuilder[BranchTerminatorBuilder]
    with NonTrainableLayerBuilder[BranchTerminatorBuilder] {

  override def repr
  : BranchTerminatorBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[BranchTerminatorBuilder]

  override protected def doCopy()
  : BranchTerminatorBuilder = BranchTerminatorBuilder()


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def weightLayoutFor(hints:   BuildHints,
                               builder: TensorLayoutBufferBuilder)
  : BuildHints = BuildHints.zero

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = BuildHints.zero

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : BranchTerminator = new BranchTerminator(this, hints, seed, weightsBuilder)

}


object BranchTerminatorBuilder {

  final def apply()
  : BranchTerminatorBuilder = new BranchTerminatorBuilder

}

final case class BranchTerminatorContext(input: Tensor)
  extends PredictContext {
}
