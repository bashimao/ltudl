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
import scala.util.hashing._

abstract class Constraint[TBuilder <: ConstraintBuilder[_]]
  extends MapLayer[TBuilder] {

  final override lazy val outputPlatform
  : Platform = inputHints.platform

  final val domain
  : TensorDomain = builder.domain

  final val scaleCoefficient
  : Real = builder.scaleCoefficient


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = {
    val scaleFactor = domain match {
      case TensorDomain.Value =>
        scaleCoefficient / input.layout.noValues
      case TensorDomain.Unit =>
        scaleCoefficient / input.layout.size.noValues
      case TensorDomain.Channel =>
        scaleCoefficient / input.layout.noTuples
      case TensorDomain.Sample =>
        scaleCoefficient / input.layout.noSamples
      case TensorDomain.Batch =>
        scaleCoefficient
      case _ =>
        throw new MatchError(domain)
    }
    (input, ConstraintPredictContext(scaleFactor))
  }

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = output


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

}

abstract class ConstraintBuilder[TThis <: ConstraintBuilder[_]]
  extends MapLayerBuilder[TThis] {

  final private var _domain
  : TensorDomain = TensorDomain.Sample

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

  /**
    * Sometimes called beta (divergence criteria)!
    */
  final var scaleCoefficient
  : Real = Real.one

  final def setScaleCoefficient(value: Real): TThis = {
    scaleCoefficient_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = _domain :: f"$scaleCoefficient%.4g" :: super.doToString()

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _domain.hashCode())
    tmp = MurmurHash3.mix(tmp, scaleCoefficient.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ConstraintBuilder[TThis] =>
      _domain          == other._domain &&
      scaleCoefficient == other.scaleCoefficient
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ConstraintBuilder[TThis] =>
        other._domain          = _domain
        other.scaleCoefficient = scaleCoefficient
      case _ =>
    }
  }

  final override def outputPlatformFor(hints: BuildHints)
  : Platform = hints.platform

}

abstract class ConstraintEx[TBuilder <: ConstraintExBuilder[_]]
  extends Constraint[TBuilder]
    with NonTrainableLayer[TBuilder] {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doEvaluate(input:     Tensor,
                                          reference: Tensor,
                                          output:    Tensor,
                                          context:   PredictContext)
  : Real = context match {
    case ConstraintPredictContext(scaleFactor) =>
      doEvaluate(reference, output, scaleFactor)
    case _ =>
      throw new MatchError(context)
  }

  protected def doEvaluate(reference: Tensor, output: Tensor, scaleFactor: Real)
  : Real


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override def backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.Required

  final override protected def doDeriveInputError(input:     Tensor,
                                                  reference: Tensor,
                                                  output:    Tensor,
                                                  context:   PredictContext,
                                                  error:     Tensor)
  : Tensor = context match {
    case ConstraintPredictContext(scaleFactor) =>
      doDeriveInputError(reference, output, scaleFactor, error)
    case _ =>
      throw new MatchError(context)
  }

  protected def doDeriveInputError(reference:   Tensor,
                                   output:      Tensor,
                                   scaleFactor: Real,
                                   error:       Tensor)
  : Tensor

}

abstract class ConstraintExBuilder[TThis <: ConstraintExBuilder[_]]
  extends ConstraintBuilder[TThis]
    with NonTrainableLayerBuilder[TThis] {

  // ---------------------------------------------------------------------------
  //    Binding / weights related
  // ---------------------------------------------------------------------------
  final override protected def doWeightLayoutFor(hints:   BuildHints,
                                                 builder: TensorLayoutBufferBuilder)
  : Unit = {}

}

final case class ConstraintPredictContext(scaleFactor: Real)
  extends PredictContext {
}