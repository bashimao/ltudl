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

package edu.latrobe.blaze.objectives

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.time._
import scala.util.hashing._

/**
  * With this you can add in fancy conditions that depend on other stuff.
  */
final class CustomObjective(override val builder: CustomObjectiveBuilder,
                            override val seed:    InstanceSeed)
  extends IndependentObjective[CustomObjectiveBuilder] {

  val callbackFn
  : () => ObjectiveEvaluationResult = builder.callbackFn

  override protected def doEvaluate(sink:                Sink,
                                    optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : Option[ObjectiveEvaluationResult] = Option(callbackFn())

}

final class CustomObjectiveBuilder
  extends IndependentObjectiveBuilder[CustomObjectiveBuilder] {

  override def repr
  : CustomObjectiveBuilder = this

  private var _callbackFn
  : () => ObjectiveEvaluationResult = () => null

  def callbackFn
  : () => ObjectiveEvaluationResult = _callbackFn

  def callbackFn_=(value: () => ObjectiveEvaluationResult)
  : Unit = {
    require(value != null)
    _callbackFn = value
  }

  def setCallbackFn(value: () => ObjectiveEvaluationResult)
  : CustomObjectiveBuilder = {
    callbackFn_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _callbackFn :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _callbackFn.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[CustomObjectiveBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: CustomObjectiveBuilder =>
      _callbackFn == other._callbackFn
    case _ =>
      false
  })

  override protected def doCopy()
  : CustomObjectiveBuilder = CustomObjectiveBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: CustomObjectiveBuilder =>
        other._callbackFn = _callbackFn
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : CustomObjective = new CustomObjective(this, seed)

}

object CustomObjectiveBuilder {

  final def apply()
  : CustomObjectiveBuilder = new CustomObjectiveBuilder

  final def apply(callbackFn: () => ObjectiveEvaluationResult)
  : CustomObjectiveBuilder = apply().setCallbackFn(callbackFn)

}