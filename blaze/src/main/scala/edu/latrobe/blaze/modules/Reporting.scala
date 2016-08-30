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

import scala.util.hashing.MurmurHash3

abstract class ReportingLayer[TBuilder <: ReportingLayerBuilder[_]]
  extends NonTrainableMapLayer[TBuilder]
    with NonPenalizing {

  final val reportingPhase
  : ReportingPhase = builder.reportingPhase

  final val reportingInterval
  : Long = builder.reportingInterval

  final override lazy val outputPlatform
  : Platform = inputHints.platform

  private var iterationNo: Long = 0L


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  final override protected def doPredict(mode:           Mode,
                                         inPlaceAllowed: Boolean,
                                         input:          Tensor,
                                         reference:      Tensor)
  : (Tensor, PredictContext) = {
    reportingPhase match {
      case ReportingPhase.Forward =>
        if (iterationNo % reportingInterval == 0L) {
          doPredict(input)
        }
        iterationNo += 1L
      case _ =>
    }
    (input, EmptyContext)
  }

  protected def doPredict(input: Tensor)
  : Unit

  final override protected def doPredictInv(output:  Tensor,
                                            context: PredictContext)
  : Tensor = output


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
  : Tensor = {
    reportingPhase match {
      case ReportingPhase.Backward =>
        if (iterationNo % reportingInterval == 0L) {
          doDeriveInputError(error)
        }
        iterationNo += 1L
      case _ =>
    }
    error
  }

  protected def doDeriveInputError(error: Tensor)
  : Unit

}

abstract class ReportingLayerBuilder[TThis <: ReportingLayerBuilder[_]]
  extends NonTrainableMapLayerBuilder[TThis] {

  private var _reportingPhase
  : ReportingPhase = ReportingPhase.Forward

  final def reportingPhase: ReportingPhase = _reportingPhase

  final def reportingPhase_=(value: ReportingPhase)
  : Unit = {
    require(value != null)
    _reportingPhase = value
  }

  final def setReportingPhase(value: ReportingPhase)
  : TThis = {
    reportingPhase_=(value)
    repr
  }

  final private var _reportingInterval
  : Long = 1L

  final def reportingInterval
  : Long = _reportingInterval

  final def reportingInterval_=(value: Long)
  : Unit = {
    require(value > 0L)
    _reportingInterval = value
  }

  final def setReportingInterval(value: Long)
  : TThis = {
    reportingInterval_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = _reportingPhase :: _reportingInterval :: super.doToString()

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _reportingPhase.hashCode())
    tmp = MurmurHash3.mix(tmp, _reportingInterval.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ReportingLayerBuilder[TThis] =>
      _reportingPhase    == other._reportingPhase &&
      _reportingInterval == other._reportingInterval
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ReportingLayerBuilder[TThis] =>
        other._reportingPhase    = _reportingPhase
        other._reportingInterval = _reportingInterval
      case _ =>
    }
  }

  final override def outputPlatformFor(hints: BuildHints)
  : Platform = hints.platform

}

abstract class ReportingPhase
  extends Serializable

object ReportingPhase {

  case object Forward
    extends ReportingPhase

  case object Backward
    extends ReportingPhase

}