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
import scala.collection._

final class ComplexExitCode(val results: Set[OptimizerExitCode])
  extends OptimizerExitCode {

  override def toString
  : String = s"ComplexExitCode[$results]"

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ComplexExitCode]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ComplexExitCode =>
      results == other.results
    case _ =>
      false
  })

  override def description
  : String = {
    s"${results.size} distinct diverging exit conditions have been observed!"
  }

  override def indicatesConvergence
  : Boolean = results.forall(_.indicatesConvergence)

  override def indicatesFailure
  : Boolean = results.exists(_.indicatesFailure)

  override def +(other: OptimizerExitCode)
  : ComplexExitCode = {
    val newSet = other match {
      case other: ComplexExitCode =>
        results ++ other.results
      case _ =>
        results + other
    }
    ComplexExitCode(newSet)
  }

}

object ComplexExitCode {

  final def apply(results: Set[OptimizerExitCode])
  : ComplexExitCode = new ComplexExitCode(results)

}
