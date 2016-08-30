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
import edu.latrobe.sizes._
import scala.util.hashing._

abstract class AggregationLayer[TBuilder <: AggregationLayerBuilder[_]]
  extends Layer[TBuilder]
    with NonTrainableLayer[TBuilder]
    with NonPenalizing {

  final override val outputHints
  : BuildHints = builder.outputHintsFor(inputHints)

  final val domain
  : TensorDomain = builder.domain


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = {
    val out = domain match {
      case TensorDomain.Unit =>
        doPredictPerUnit(input)

      case TensorDomain.Channel =>
        doPredictPerChannel(input)

      case TensorDomain.Sample =>
        doPredictPerSample(input)

      case TensorDomain.Batch =>
        doPredictPerBatch(input)

      case _ =>
        throw new MatchError(domain)
    }
    (out, AggregationLayerContext(input.layout))
  }

  protected def doPredictPerUnit(input: Tensor)
  : Tensor

  protected def doPredictPerChannel(input: Tensor)
  : Tensor

  protected def doPredictPerSample(input: Tensor)
  : Tensor

  protected def doPredictPerBatch(input: Tensor)
  : Tensor

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = throw new UnsupportedOperationException


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
    case AggregationLayerContext(inputLayout) =>
      domain match {
        case TensorDomain.Unit =>
          doDeriveInputErrorPerUnit(inputLayout, error)

        case TensorDomain.Channel =>
          ddoDeriveInputErrorPerChannel(inputLayout, error)

        case TensorDomain.Sample =>
          doDeriveInputErrorPerSample(inputLayout, error)

        case TensorDomain.Batch =>
          doDeriveInputErrorPerBatch(inputLayout, error)

        case _ =>
          throw new MatchError(domain)
      }
    case _ =>
      throw new MatchError(context)
  }

  protected def doDeriveInputErrorPerUnit(inputLayout: TensorLayout,
                                          error:       Tensor)
  : Tensor

  protected def ddoDeriveInputErrorPerChannel(inputLayout: TensorLayout,
                                              error:       Tensor)
  : Tensor

  protected def doDeriveInputErrorPerSample(inputLayout: TensorLayout,
                                            error:       Tensor)
  : Tensor

  protected def doDeriveInputErrorPerBatch(inputLayout: TensorLayout,
                                           error:       Tensor)
  : Tensor

}

abstract class AggregationLayerBuilder[TThis <: AggregationLayerBuilder[_]]
  extends LayerBuilder[TThis]
    with NonTrainableLayerBuilder[TThis] {

  final private var _domain
  : TensorDomain = TensorDomain.Unit

  final def domain
  : TensorDomain = _domain

  final def domain_=(value: TensorDomain): Unit = {
    require(value != null)
    _domain = value
  }

  final def setDomain(value: TensorDomain)
  : TThis = {
    domain_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = _domain :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _domain.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: AggregationLayerBuilder[TThis] =>
      _domain == other._domain
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder): Unit = {
    super.copyTo(other)
    other match {
      case other: AggregationLayerBuilder[TThis] =>
        other._domain = _domain
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  final override def weightLayoutFor(hints:   BuildHints,
                                     builder: TensorLayoutBufferBuilder)
  : BuildHints = outputHintsFor(hints)

  def outputPlatformFor(hints: BuildHints)
  : Platform

  final def outputLayoutFor(layoutHint: TensorLayout)
  : IndependentTensorLayout = _domain match {
    case TensorDomain.Unit =>
      layoutHint.derive(1)
    case TensorDomain.Channel =>
      IndependentTensorLayout.derive(layoutHint.size.noChannels)
    case TensorDomain.Sample =>
      layoutHint.derive(Size1.one)
    case TensorDomain.Batch =>
      IndependentTensorLayout.one
    case _ =>
      throw new MatchError(_domain)
  }

  final override def outputHintsFor(hints: BuildHints)
  : BuildHints = {
    val platform = outputPlatformFor(hints)
    val layout   = outputLayoutFor(hints.layout)
    hints.derive(platform, layout)
  }

}

final case class AggregationLayerContext(inputLayout: TensorLayout)
  extends PredictContext {
}