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

package edu.latrobe.blaze.regularizers

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.parameters._
import java.util.UUID
import scala.collection._
import scala.util.hashing._

/**
 * A constraint that penalizes on the weights.
 *
 * More or less just a layer that allows you to bind regularizers at global,
 * weightsGroup or segment level.
 */
abstract class WeightDecay[TBuilder <: WeightDecayBuilder[_]]
  extends SimpleRegularizer[TBuilder] {

  final val scaleCoefficient
  : Parameter = builder.scaleCoefficient.build(
    builder.scaleCoefficientLabel,
    seed
  )

  final val scaleDomain
  : TensorDomain = builder.scaleDomain

  override val parameters
  : Map[UUID, Parameter] = super.parameters + scaleCoefficient.toTuple

  override protected def doClose()
  : Unit = {
    scaleCoefficient.close()
    super.doClose()
  }

  final def scaleFactorFor(phaseNo:     Long,
                           inputLayout: TensorLayout)
  : Real = {
    val sf = scaleCoefficient.get(phaseNo)
    scaleDomain match {
      case TensorDomain.Value =>
        sf / inputLayout.noValues
      case TensorDomain.Unit =>
        sf / inputLayout.size.noValues
      case TensorDomain.Channel =>
        sf / inputLayout.noTuples
      case TensorDomain.Sample =>
        sf / inputLayout.noSamples
      case TensorDomain.Batch =>
        sf
      case _ =>
        throw new MatchError(scaleDomain)
    }
  }


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override def evaluate(phaseNo:   Long,
                              weights:   ValueTensorBuffer,
                              input:     Tensor,
                              reference: Tensor,
                              output:    Tensor)
  : Real = {
    val sf = scaleFactorFor(phaseNo, input.layout)
    val w  = baseScope.map(
      weights.createIntersectionView
    ).getOrElse(weights)

    val result = w.foldLeftSegments(
      Real.zero
    )(_ + doEvaluate(_))
    result * sf
  }

  protected def doEvaluate(weights: ValueTensor)
  : Real


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override def deriveGradients(phaseNo:   Long,
                                     weights:   ValueTensorBuffer,
                                     input:     Tensor,
                                     reference: Tensor,
                                     output:    Tensor,
                                     sink:      ValueTensorBuffer)
  : Unit = {
    // Figure out current intersection of buffers.
    val sf = scaleFactorFor(phaseNo, input.layout)
    val w  = baseScope.map(
      weights.createIntersectionView
    ).getOrElse(weights)
    val s  = baseScope.map(
      sink.createIntersectionView
    ).getOrElse(sink)

    // Only process further if a scope is present for the currently selected
    // sink.
    s.foreachSegment(
      w
    )(doDeriveWeightGradients(_, sf, _))
  }

  protected def doDeriveWeightGradients(sink:        ValueTensor,
                                        scaleFactor: Real,
                                        weights:     ValueTensor)
  : Unit


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : RegularizerState = WeightDecayState(super.state, scaleCoefficient.state)

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: WeightDecayState =>
        scaleCoefficient.restoreState(state.scaleCoefficient)
      case _ =>
        throw new MatchError(state)
    }
  }

}

abstract class WeightDecayBuilder[TThis <: WeightDecayBuilder[_]]
  extends SimpleRegularizerBuilder[TThis] {

  /**
    * Also called lambda.
    */
  final private var _scaleCoefficient
  : ParameterBuilder = ConstantValueBuilder.one

  final def scaleCoefficient
  : ParameterBuilder = _scaleCoefficient

  final def scaleCoefficient_=(value: ParameterBuilder)
  : Unit = {
    require(value != null)
    _scaleCoefficient = value
  }

  final def setScaleCoefficient(value: ParameterBuilder)
  : TThis = {
    scaleCoefficient_=(value)
    repr
  }

  /**
    * Only for printing.
    */
  final private var _scaleCoefficientLabel
  : String = "lambda"

  final def scaleCoefficientLabel
  : String = _scaleCoefficientLabel

  final def scaleCoefficientLabel_=(value: String)
  : Unit = {
    require(value != null)
    _scaleCoefficientLabel = value
  }

  final def setScaleCoefficientLabel(value: String)
  : TThis = {
    scaleCoefficientLabel_=(value)
    repr
  }

  final private var _scaleDomain
  : TensorDomain = TensorDomain.Batch

  final def scaleDomain
  : TensorDomain = _scaleDomain

  final def scaleDomain_=(value: TensorDomain)
  : Unit = {
    require(value != null)
    _scaleDomain = value
  }

  final def setScaleDomain(value: TensorDomain)
  : TThis = {
    scaleDomain_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = {
    _scaleCoefficient :: _scaleCoefficientLabel :: _scaleDomain :: super.doToString()
  }

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _scaleCoefficient.hashCode())
    tmp = MurmurHash3.mix(tmp, _scaleCoefficientLabel.hashCode())
    tmp = MurmurHash3.mix(tmp, _scaleDomain.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: WeightDecayBuilder[_] =>
      _scaleCoefficient      == other._scaleCoefficient      &&
      _scaleCoefficientLabel == other._scaleCoefficientLabel &&
      _scaleDomain           == other._scaleDomain
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: WeightDecayBuilder[_] =>
        other._scaleCoefficient      = _scaleCoefficient.copy
        other._scaleCoefficientLabel = _scaleCoefficientLabel
        other._scaleDomain           = _scaleDomain
      case _ =>
    }
  }

  //----------------------------------------------------------------------------
  //   Recursive mutable state related.
  //----------------------------------------------------------------------------
  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    _scaleCoefficient.permuteSeeds(fn)
  }

}

final case class WeightDecayState(override val parent: RegularizerState,
                                  scaleCoefficient:    InstanceState)
  extends RegularizerState
