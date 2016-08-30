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

abstract class Combiner[TBuilder <: CombinerBuilder[_]]
  extends Layer[TBuilder]
    with NonTrainableLayer[TBuilder]
    with NonPenalizing {

  final override val outputHints
  : BuildHints = builder.outputHintsFor(inputHints)


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = input match {
    case input: TensorTable =>
      val result = doPredict(input)
      (result, CombinerContext(input.layout))
    case _ =>
      (input, EmptyContext)
  }

  protected def doPredict(input: TensorTable)
  : Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = context match {
    case CombinerContext(inpLayout) =>
      val out = output
      val inp = TensorTable(inpLayout.map(out.createSiblingAndClear))
      doPredictInv(out, inp)
      inp
    case _ =>
      output
  }

  protected def doPredictInv(output: Tensor,
                             input:  TensorTable)
  : Unit


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = context match {
    case CombinerContext(inpLayout) =>
      val oldErr = error
      val newErr = TensorTable(inpLayout.map(oldErr.createSiblingAndClear))
      doDeriveInputError(
        input, reference, output, oldErr, newErr
      )
      newErr
    case _ =>
      error
  }

  protected def doDeriveInputError(input:     Tensor,
                                   reference: Tensor,
                                   output:    Tensor,
                                   oldError:  Tensor,
                                   newError:  TensorTable)
  : Unit

}

abstract class CombinerBuilder[TThis <: CombinerBuilder[_]]
  extends LayerBuilder[TThis]
    with NonTrainableLayerBuilder[TThis] {

  final override def weightLayoutFor(hints:   BuildHints,
                                     builder: TensorLayoutBufferBuilder)
  : BuildHints = outputHintsFor(hints)

  final def outputPlatformFor(platformHint: Platform)
  : Platform = platformHint match {
    case platformHint: PlatformTable =>
      outputPlatformFor(platformHint)
    case _ =>
      platformHint
  }

  def outputPlatformFor(platformHint: PlatformTable)
  : Platform

  final def outputLayoutFor(layoutHint: TensorLayout)
  : TensorLayout = layoutHint match {
    case layoutHint: TensorLayoutTable =>
      outputLayoutFor(layoutHint)
    case _ =>
      layoutHint
  }

  def outputLayoutFor(layoutHint: TensorLayoutTable)
  : TensorLayout

  final override def outputHintsFor(hints: BuildHints)
  : BuildHints = {
    val platform = outputPlatformFor(hints.platform)
    val layout   = outputLayoutFor(hints.layout)
    hints.derive(platform, layout)
  }

}

final case class CombinerContext(inputLayout: TensorLayoutTable)
  extends PredictContext {
}
