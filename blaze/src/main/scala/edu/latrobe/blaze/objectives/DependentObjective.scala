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

abstract class DependentObjective[TBuilder <: DependentObjectiveBuilder[_]]
  extends ObjectiveEx[TBuilder] {

  final protected val children
  : Array[Objective] = {
    builder.children.map(
      _.build(seed)
    ).toArray
  }

  override protected def doClose()
  : Unit = {
    ArrayEx.foreach(
      children
    )(_.close())
    super.doClose()
  }


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : ObjectiveState = DependentObjectiveState(
    super.state,
    ArrayEx.map(
      children
    )(_.state)
  )

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: DependentObjectiveState =>
        ArrayEx.foreach(
          children,
          state.children
        )(_.restoreState(_))
      case _ =>
        throw new MatchError(state)
    }
  }

}

abstract class DependentObjectiveBuilder[TThis <: DependentObjectiveBuilder[_]]
  extends ObjectiveExBuilder[TThis] {

  final val children
  : mutable.Buffer[ObjectiveBuilder] = mutable.Buffer.empty

  final def +=(objective: ObjectiveBuilder)
  : TThis = {
    children += objective
    repr
  }

  final def ++=(objectives: TraversableOnce[ObjectiveBuilder])
  : TThis = {
    children ++= objectives
    repr
  }


  override protected def doToString()
  : List[Any] = children.length :: super.doToString()


  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), children.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: DependentObjectiveBuilder[_] =>
      children == other.children
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: DependentObjectiveBuilder[_] =>
        other.children.clear()
        other.children ++= children.map(_.copy)
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //   Cascading mutable state.
  // ---------------------------------------------------------------------------
  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    children.foreach(
      _.permuteSeeds(fn)
    )
  }

}

final case class DependentObjectiveState(override val parent: InstanceState,
                                         children:            Array[InstanceState])
  extends ObjectiveState {
}


abstract class DependentObjectiveEx[TBuilder <: DependentObjectiveExBuilder[_]]
  extends DependentObjective[TBuilder] {

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
    ArrayEx.foreach(children)(child => {
      val result = child.evaluate(
        sink,
        optimizer, runBeginIterationNo, runBeginTime, runNoSamples,
        model,
        batch, output, value
      )

      if (result.isDefined) {
        return result
      }
    })
    None
  }

}

abstract class DependentObjectiveExBuilder[TThis <: DependentObjectiveExBuilder[_]]
  extends DependentObjectiveBuilder[TThis] {
}
