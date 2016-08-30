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

package edu.latrobe.blaze.objectives

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.time._
import scala.util.hashing._

/**
 * If absolute cost drops below the target.
 */
final class ValueLimit(override val builder: ValueLimitBuilder,
                       override val seed:    InstanceSeed)
  extends BinaryTriggerObjective[ValueLimitBuilder](builder.result) {
  require(builder != null && seed != null)

  val range
  : RealRange = builder.range

  override protected def doEvaluate(optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : Boolean = range.contains(value)

}

final class ValueLimitBuilder
  extends BinaryTriggerObjectiveBuilder[ValueLimitBuilder] {

  override def repr
  : ValueLimitBuilder = this

  private var _range
  : RealRange = RealRange(Real.negativeInfinity, Real.epsilon)

  def range
  : RealRange = _range

  def range_=(value: RealRange): Unit = {
    require(value != null)
    _range = value
  }

  def setRange(value: RealRange)
  : ValueLimitBuilder = {
    range_=(value)
    this
  }

  private var _result
  : ObjectiveEvaluationResult = ObjectiveEvaluationResult.Convergence

  def result
  : ObjectiveEvaluationResult = _result

  def result_=(value: ObjectiveEvaluationResult)
  : Unit = {
    require(value != null)
    _result = value
  }

  def setResult(value: ObjectiveEvaluationResult)
  : ValueLimitBuilder = {
    result_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _range :: _result :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _range.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ValueLimitBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ValueLimitBuilder =>
      _range  == other._range &&
      _result == other._result
    case _ =>
      false
  })


  override protected def doCopy()
  : ValueLimitBuilder = ValueLimitBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ValueLimitBuilder =>
        other._range  = _range
        other._result = _result
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : ValueLimit = new ValueLimit(this, seed)

}

object ValueLimitBuilder {

  final def apply()
  : ValueLimitBuilder = new ValueLimitBuilder

  final def apply(thresholdMin: Real, thresholdMax: Real)
  : ValueLimitBuilder = apply(RealRange(thresholdMin, thresholdMax))

  final def apply(threshold: RealRange)
  : ValueLimitBuilder = apply().setRange(threshold)

  final def apply(threshold: RealRange,
                  result:    ObjectiveEvaluationResult)
  : ValueLimitBuilder = apply(threshold).setResult(result)

}