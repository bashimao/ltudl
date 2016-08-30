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

package edu.latrobe.blaze.optimizerexitcodes

import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

/**
  * The optimization process was aborted because an objective has been met.
  */
final class ObjectiveReached(val result: ObjectiveEvaluationResult)
  extends IndependentOptimizerExitCode {

  override def toString
  : String = s"ObjectiveReached[$result]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), result.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ObjectiveReached]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ObjectiveReached =>
      result == other.result
    case _ =>
      false
  })

  override def description
  : String = s"An objectives evaluated non-nul. Outcome: $result"

  override def indicatesConvergence
  : Boolean = result match {
    case ObjectiveEvaluationResult.Convergence =>
      true
    case _ =>
      false
  }

  override def indicatesFailure
  : Boolean = result match {
    case ObjectiveEvaluationResult.Failure =>
      true
    case _ =>
      false
  }

}

object ObjectiveReached {

  final def apply(evaluationResult: ObjectiveEvaluationResult)
  : ObjectiveReached = new ObjectiveReached(evaluationResult)

}
