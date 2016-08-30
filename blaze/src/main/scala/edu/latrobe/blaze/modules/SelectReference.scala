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
import edu.latrobe.blaze.TensorDependency._
import edu.latrobe.io.graph.{Edge, LineStyle, Node}

import scala.collection.mutable

/**
 * This module will simply transfer the reference tensor to the output. Backprop
 * through this layer is - of course - impossible! Only useful as a constraint
 * selector.
 */
final class SelectReference(override val builder:        SelectReferenceBuilder,
                            override val inputHints:     BuildHints,
                            override val seed:           InstanceSeed,
                            override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Layer[SelectReferenceBuilder]
    with NonTrainableLayer[SelectReferenceBuilder]
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
  : (Tensor, PredictContext) = (reference, EmptyContext)

  override protected def doPredictInv(output: Tensor, context: PredictContext)
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
  : Tensor = throw new UnsupportedOperationException

}

final class SelectReferenceBuilder
  extends LayerBuilder[SelectReferenceBuilder]
    with NonTrainableLayerBuilder[SelectReferenceBuilder] {

  override def repr
  : SelectReferenceBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[SelectReferenceBuilder]

  override protected def doCopy()
  : SelectReferenceBuilder = SelectReferenceBuilder()


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def weightLayoutFor(hints:   BuildHints,
                               builder: TensorLayoutBufferBuilder)
  : BuildHints = outputHintsFor(hints)

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = hints.derive(hints.referencePlatform, hints.referenceLayout)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : SelectReference = new SelectReference(this, hints, seed, weightsBuilder)

}

object SelectReferenceBuilder {

  final def apply()
  : SelectReferenceBuilder = new SelectReferenceBuilder

}