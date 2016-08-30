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
import scala.collection._
import scala.util.hashing._

/**
  * Will invert the outcome of the dependent objective.
  *
  * Always returns "neutral" result!
  */
final class InvertTrigger(override val builder: InvertTriggerBuilder,
                          override val seed:    InstanceSeed)
  extends DependentObjectiveEx[InvertTriggerBuilder] {

  val result
  : Option[ObjectiveEvaluationResult] = Some(builder.result)

  override protected def doEvaluate(sink:                Sink,
                                    optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : Option[ObjectiveEvaluationResult] = {
    val tmp = super.doEvaluate(
      sink,
      optimizer, runBeginIterationNo, runBeginTime, runNoSamples,
      model,
      batch, output, value
    )
    if (tmp.isDefined) {
      None
    }
    else {
      result
    }
  }

}


final class InvertTriggerBuilder
  extends DependentObjectiveExBuilder[InvertTriggerBuilder] {

  override def repr
  : InvertTriggerBuilder = this

  private var _result
  : ObjectiveEvaluationResult = ObjectiveEvaluationResult.Neutral

  def result
  : ObjectiveEvaluationResult = _result

  def result_=(value: ObjectiveEvaluationResult)
  : Unit = {
    require(value != null)
    _result = value
  }

  def setResult(value: ObjectiveEvaluationResult)
  : InvertTriggerBuilder = {
    result_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _result :: super.doToString()

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[InvertTriggerBuilder]

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _result.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: InvertTriggerBuilder =>
      _result == other._result
    case _ =>
      false
  })

  override protected def doCopy()
  : InvertTriggerBuilder = InvertTriggerBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: InvertTriggerBuilder =>
        other._result = _result
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : InvertTrigger = new InvertTrigger(this, seed)

}

object InvertTriggerBuilder {

  final def apply()
  : InvertTriggerBuilder = new InvertTriggerBuilder

  final def apply(child0: ObjectiveBuilder)
  : InvertTriggerBuilder = apply() += child0

  final def apply(child0: ObjectiveBuilder, children: ObjectiveBuilder*)
  : InvertTriggerBuilder = apply(child0) ++= children

  final def apply(children: TraversableOnce[ObjectiveBuilder])
  : InvertTriggerBuilder = apply() ++= children

}
