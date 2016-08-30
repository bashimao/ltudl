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

import scala.util.hashing.MurmurHash3

/**
 * Modules where each input is directly connected to an output. This is true
 * for most activation functions.
 */
abstract class MapLayer[TBuilder <: MapLayerBuilder[_]]
  extends Layer[TBuilder] {

  /**
   * Override this with lazy val or def!
   */
  def outputPlatform
  : Platform

  final val outputHints
  : BuildHints = inputHints.derive(outputPlatform)

}

abstract class MapLayerBuilder[TThis <: MapLayerBuilder[_]]
  extends LayerBuilder[TThis] {

  // ---------------------------------------------------------------------------
  //    Binding / weights related
  // ---------------------------------------------------------------------------
  final override def weightLayoutFor(hints:   BuildHints,
                                     builder: TensorLayoutBufferBuilder)
  : BuildHints = {
    doWeightLayoutFor(hints, builder)
    outputHintsFor(hints)
  }

  protected def doWeightLayoutFor(hints:   BuildHints,
                                  builder: TensorLayoutBufferBuilder)
  : Unit

  def outputPlatformFor(hints: BuildHints)
  : Platform

  final def outputHintsFor(hints: BuildHints)
  : BuildHints = hints.derive(outputPlatformFor(hints))

}

/**
  * The extended map layer allows you to specify focus domains.
  */
abstract class MapLayerEx[TBuilder <: MapLayerExBuilder[_]]
  extends MapLayer[TBuilder] {

  final val domain
  : TensorDomain = builder.domain


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = domain match {
    case TensorDomain.Value =>
      require(input.layout == inputLayoutHint)
      doPredictPerValue(mode, inPlaceAllowed, input, reference)

    case TensorDomain.Unit =>
      require(input.layout.size == inputSizeHint)
      doPredictPerUnit(mode, inPlaceAllowed, input, reference)

    case TensorDomain.Channel =>
      require(input.layout.size.noChannels == inputSizeHint.noChannels)
      doPredictPerChannel(mode, inPlaceAllowed, input, reference)

    case TensorDomain.Sample =>
      require(input.layout.noSamples == inputLayoutHint.noSamples)
      doPredictPerSample(mode, inPlaceAllowed, input, reference)

    case TensorDomain.Batch =>
      doPredictPerBatch(mode, inPlaceAllowed, input, reference)

    case _ =>
      throw new MatchError(domain)
  }

  protected def doPredictPerValue(mode:           Mode,
                                  inPlaceAllowed: Boolean,
                                  input:          Tensor,
                                  reference:      Tensor)
  : (Tensor, PredictContext)

  protected def doPredictPerUnit(mode:           Mode,
                                 inPlaceAllowed: Boolean,
                                 input:          Tensor,
                                 reference:      Tensor)
  : (Tensor, PredictContext)

  protected def doPredictPerChannel(mode:           Mode,
                                    inPlaceAllowed: Boolean,
                                    input:          Tensor,
                                    reference:      Tensor)
  : (Tensor, PredictContext)

  protected def doPredictPerSample(mode:           Mode,
                                   inPlaceAllowed: Boolean,
                                   input:          Tensor,
                                   reference:      Tensor)
  : (Tensor, PredictContext)

  protected def doPredictPerBatch(mode:           Mode,
                                  inPlaceAllowed: Boolean,
                                  input:          Tensor,
                                  reference:      Tensor)
  : (Tensor, PredictContext)

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = domain match {
    case TensorDomain.Value =>
      require(output.layout == inputLayoutHint)
      doPredictInvPerValue(output, context)

    case TensorDomain.Unit =>
      require(output.layout.size == inputSizeHint)
      doPredictInvPerUnit(output, context)

    case TensorDomain.Channel =>
      require(output.layout.size.noChannels == inputSizeHint.noChannels)
      doPredictInvPerChannel(output, context)

    case TensorDomain.Sample =>
      require(output.layout.noSamples == inputLayoutHint.noSamples)
      doPredictInvPerSample(output, context)

    case TensorDomain.Batch =>
      doPredictInvPerBatch(output, context)

    case _ =>
      throw new MatchError(domain)
  }

  protected def doPredictInvPerValue(output:  Tensor,
                                     context: PredictContext)
  : Tensor

  protected def doPredictInvPerUnit(output:  Tensor,
                                    context: PredictContext)
  : Tensor

  protected def doPredictInvPerChannel(output:  Tensor,
                                       context: PredictContext)
  : Tensor

  protected def doPredictInvPerSample(output:  Tensor,
                                      context: PredictContext)
  : Tensor

  protected def doPredictInvPerBatch(output:  Tensor,
                                     context: PredictContext)
  : Tensor

}

abstract class MapLayerExBuilder[TThis <: MapLayerExBuilder[_]]
  extends MapLayerBuilder[TThis] {

  def defaultDomain()
  : TensorDomain

  final private var _domain
  : TensorDomain = defaultDomain()

  final def domain
  : TensorDomain = _domain

  final def domain_=(value: TensorDomain)
  : Unit = {
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
    case other: MapLayerExBuilder[_] =>
      _domain == other._domain
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: MapLayerExBuilder[_] =>
        other._domain = _domain
      case _ =>
    }
  }

}
