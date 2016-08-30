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

package edu.latrobe.blaze

import edu.latrobe._
import edu.latrobe.time._
import scala.collection._

/**
  * Optimization objective for an optimizer.
  *
  * Can be mutable.
  */
abstract class Objective
  extends InstanceEx[ObjectiveBuilder] {

  /**
    * The gradients are usually outdated by one iteration.
    *
    * @return Returns the objective that has been satisfied or "null" if no
    *         objective has been satisfied. The returned object can be different
    *         from the current object if a complex objective is being used.
    */
  final def evaluate(sink:                Sink,
                     optimizer:           OptimizerLike,
                     runBeginIterationNo: Long,
                     runBeginTime:        Timestamp,
                     runNoSamples:        Long,
                     model:               Module,
                     batch:               Batch,
                     output:              Tensor,
                     value:               Real)
  : Option[ObjectiveEvaluationResult] = {
    val result = doEvaluate(
      sink,
      optimizer, runBeginIterationNo, runBeginTime, runNoSamples,
      model,
      batch, output, value
    )
    result.foreach(result => {
      if (logger.isTraceEnabled) {
        logger.trace(s"Objective $toString evaluated as $result")
      }
    })
    result
  }

  protected def doEvaluate(sink:                Sink,
                           optimizer:           OptimizerLike,
                           runBeginIterationNo: Long,
                           runBeginTime:        Timestamp,
                           runNoSamples:        Long,
                           model:               Module,
                           batch:               Batch,
                           output:              Tensor,
                           value:               Real)
  : Option[ObjectiveEvaluationResult]


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : ObjectiveState = ObjectiveStateEx(super.state)

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: ObjectiveStateEx =>
      case _ =>
        throw new MatchError(state)
    }
  }

}

/**
  * Descriptive part of an optimization objective. Must be immutable.
  */
abstract class ObjectiveBuilder
  extends InstanceExBuilder0[ObjectiveBuilder, Objective] {
}

abstract class ObjectiveState
  extends InstanceState {
}

final case class ObjectiveStateEx(override val parent: InstanceState)
  extends ObjectiveState {
}


abstract class ObjectiveEx[TBuilder <: ObjectiveExBuilder[_]]
  extends Objective {

  override def builder
  : TBuilder

}

abstract class ObjectiveExBuilder[TThis <: ObjectiveExBuilder[_]]
  extends ObjectiveBuilder {

  override def repr
  : TThis

  override protected def doCopy()
  : TThis

}

abstract class ObjectiveEvaluationResult
  extends Serializable {

  def count
  : Long

  def toMap
  : Map[ObjectiveEvaluationResult, Long]

  def +(other: ObjectiveEvaluationResult)
  : ObjectiveEvaluationResult

}

object ObjectiveEvaluationResult {

  case object Convergence
    extends ObjectiveEvaluationResult {

    override def count
    : Long = 1L

    override def toMap
    : Map[ObjectiveEvaluationResult, Long] = {
      Map[ObjectiveEvaluationResult, Long](Tuple2(Convergence, 1L))
    }

    override def +(other: ObjectiveEvaluationResult)
    : Complex = {
      val otherMap = other.toMap
      val newCount = otherMap.getOrElse(Convergence, 0L) + 1L
      Complex(otherMap + Tuple2(Convergence, newCount))
    }

  }

  case object Failure
    extends ObjectiveEvaluationResult {

    override def count
    : Long = 1L

    override def toMap
    : Map[ObjectiveEvaluationResult, Long] = {
      Map[ObjectiveEvaluationResult, Long](Tuple2(Failure, 1L))
    }

    override def +(other: ObjectiveEvaluationResult)
    : Complex = {
      val otherMap = other.toMap
      val newCount = otherMap.getOrElse(Failure, 0L) + 1L
      Complex(otherMap + Tuple2(Failure, newCount))
    }

  }

  case object Neutral
    extends ObjectiveEvaluationResult {

    override def count
    : Long = 1L

    override def toMap
    : Map[ObjectiveEvaluationResult, Long] = {
      Map[ObjectiveEvaluationResult, Long](Tuple2(Neutral, 1L))
    }

    override def +(other: ObjectiveEvaluationResult)
    : Complex = {
      val otherMap = other.toMap
      val newCount = otherMap.getOrElse(Neutral, 0L) + 1L
      Complex(otherMap + Tuple2(Neutral, newCount))
    }

  }

  case class Complex(override val toMap: Map[ObjectiveEvaluationResult, Long])
    extends ObjectiveEvaluationResult {

    override def count
    : Long = MapEx.foldLeftValues(
      0L,
      toMap
    )(_ + _)

    override def +(other: ObjectiveEvaluationResult)
    : Complex = {
      val newMap = other match {
        case Complex(otherToMap) =>
          MapEx.zipValuesEx(
            toMap,
            otherToMap
          )((a, b) => a + b, a => a, b => b)
        case _ =>
          val newCount = toMap.getOrElse(other, 0L) + 1L
          toMap + Tuple2(other, newCount)
      }
      Complex(newMap)
    }

  }

}
