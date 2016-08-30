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
import edu.latrobe.time._
import scala.collection._

trait NonTrainableLayerLike[TBuilder <: LayerBuilder[_]]
  extends Layer[TBuilder] {

  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveGradients(input:     Tensor,
                                                 reference: Tensor,
                                                 output:    Tensor,
                                                 context:   PredictContext,
                                                 error:     NextError,
                                                 sink:      ValueTensorBuffer)
  : NextError = DependentError(error, oldErr => {
    val clock = if (logger.isTraceEnabled) Stopwatch() else null

    val newErr = doDeriveInputError(
      input,
      reference,
      output,
      context,
      oldErr
    )

    if (clock != null) {
      logger.trace(
        f"$clock%s => deriveInputError(${oldErr.platform}%-4s) => $this%s"
      )
    }
    newErr
  })

  protected def doDeriveInputError(input:     Tensor,
                                   reference: Tensor,
                                   output:    Tensor,
                                   context:   PredictContext,
                                   error:     Tensor)
  : Tensor

}

/**
  * A layer that does whatever it does without weights.
  * (As opposed to containers and neural layers.)
  */
trait NonTrainableLayer[TBuilder <: LayerBuilder[_]]
  extends Layer[TBuilder]
    with NonTrainableLayerLike[TBuilder] {

  // ---------------------------------------------------------------------------
  //    Statistics.
  // ---------------------------------------------------------------------------
  final override lazy val noNeurons
  : Long = 0L


  // ---------------------------------------------------------------------------
  //    Weights related.
  // ---------------------------------------------------------------------------
  final override val weightReferences
  : Set[LabeledBufferReference] = Set.empty

  final override def reset(initializer: Initializer)
  : Unit = {}

  /**
    * Synchronizes any native hardware buffers and JVM buffers.
    */
  final override def refresh()
  : Unit = {}

  /*
  final override protected def doWeights(builder: ParameterBufferBuilder)
  : Unit = {}
  */

  final override def extractWeightsFor(neuronNo: Long)
  : Array[Real] = throw new IndexOutOfBoundsException

}

trait NonTrainableLayerBuilder[TThis <: NonTrainableLayerBuilder[_]]
  extends LayerBuilder[TThis] {

  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  final override protected def doPermuteWeightReferences(fn: LabeledBufferReference => LabeledBufferReference)
  : Unit = {}

}

abstract class NonTrainableMapLayer[TBuilder <: NonTrainableMapLayerBuilder[_]]
  extends MapLayer[TBuilder]
    with NonTrainableLayer[TBuilder] {
}

abstract class NonTrainableMapLayerBuilder[TThis <: NonTrainableMapLayerBuilder[_]]
  extends MapLayerBuilder[TThis]
    with NonTrainableLayerBuilder[TThis] {

  // ---------------------------------------------------------------------------
  //    Binding / weights related
  // ---------------------------------------------------------------------------
  final override protected def doWeightLayoutFor(hints:   BuildHints,
                                                 builder: TensorLayoutBufferBuilder)
  : Unit = {}

}

abstract class NonTrainableMapLayerEx[TBuilder <: NonTrainableMapLayerExBuilder[_]]
  extends MapLayerEx[TBuilder]
    with NonTrainableLayer[TBuilder] {

  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = domain match {
    case TensorDomain.Value =>
      require(error.layout == inputLayoutHint)
      doDeriveInputErrorPerValue(input, reference, output, context, error)

    case TensorDomain.Unit =>
      require(error.layout.size == inputSizeHint)
      doDeriveInputErrorPerUnit(input, reference, output, context, error)

    case TensorDomain.Channel =>
      require(error.layout.size.noChannels == inputSizeHint.noChannels)
      doDeriveInputErrorPerChannel(input, reference, output, context, error)

    case TensorDomain.Sample =>
      require(error.layout.noSamples == inputLayoutHint.noSamples)
      doDeriveInputErrorPerSample(input, reference, output, context, error)

    case TensorDomain.Batch =>
      doDeriveInputErrorPerBatch(input, reference, output, context, error)

    case _ =>
      throw new MatchError(domain)
  }

  protected def doDeriveInputErrorPerValue(input:     Tensor,
                                           reference: Tensor,
                                           output:    Tensor,
                                           context:   PredictContext,
                                           error:     Tensor)
  : Tensor

  protected def doDeriveInputErrorPerUnit(input:     Tensor,
                                          reference: Tensor,
                                          output:    Tensor,
                                          context:   PredictContext,
                                          error:     Tensor)
  : Tensor

  protected def doDeriveInputErrorPerChannel(input:     Tensor,
                                             reference: Tensor,
                                             output:    Tensor,
                                             context:   PredictContext,
                                             error:     Tensor)
  : Tensor

  protected def doDeriveInputErrorPerSample(input:     Tensor,
                                            reference: Tensor,
                                            output:    Tensor,
                                            context:   PredictContext,
                                            error:     Tensor)
  : Tensor

  protected def doDeriveInputErrorPerBatch(input:     Tensor,
                                           reference: Tensor,
                                           output:    Tensor,
                                           context:   PredictContext,
                                           error:     Tensor)
  : Tensor

}

abstract class NonTrainableMapLayerExBuilder[TThis <: NonTrainableMapLayerExBuilder[_]]
  extends MapLayerExBuilder[TThis]
    with NonTrainableLayerBuilder[TThis] {

  // ---------------------------------------------------------------------------
  //    Binding / weights related
  // ---------------------------------------------------------------------------
  final override protected def doWeightLayoutFor(hints:   BuildHints,
                                                 builder: TensorLayoutBufferBuilder)
  : Unit = {}

}