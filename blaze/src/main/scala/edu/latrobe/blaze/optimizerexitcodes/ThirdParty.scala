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

import scala.util.hashing.MurmurHash3

/**
  * For breeze:
  *
  * FirstOrderMinimizer.FunctionValuesConverged
  * or
  * FirstOrderMinimizer.GradientConverged
  */
final class ThirdParty(override val description:          String,
                       override val indicatesConvergence: Boolean,
                       override val indicatesFailure:     Boolean)
  extends IndependentOptimizerExitCode {

  override def toString
  : String = {
    s"ThirdParty[$description, $indicatesConvergence, $indicatesFailure]"
  }

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, description.hashCode)
    tmp = MurmurHash3.mix(tmp, indicatesConvergence.hashCode)
    tmp = MurmurHash3.mix(tmp, indicatesFailure.hashCode)
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ThirdParty]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ThirdParty =>
      description          == other.description          &&
      indicatesConvergence == other.indicatesConvergence &&
      indicatesFailure     == other.indicatesFailure
    case _ =>
      false
  })

}

object ThirdParty {

  final def apply(description:          String,
                  indicatesConvergence: Boolean,
                  indicatesFailure:     Boolean)
  : ThirdParty = new ThirdParty(
    description,
    indicatesConvergence,
    indicatesFailure
  )

  final def convergence(description: String)
  : ThirdParty = apply(
    description,
    indicatesConvergence = true,
    indicatesFailure = false
  )

  final def failure(description: String)
  : ThirdParty = apply(
    description,
    indicatesConvergence = false,
    indicatesFailure = true
  )

  final def neutral(description: String)
  : ThirdParty = apply(
    description,
    indicatesConvergence = false,
    indicatesFailure = false
  )

}
