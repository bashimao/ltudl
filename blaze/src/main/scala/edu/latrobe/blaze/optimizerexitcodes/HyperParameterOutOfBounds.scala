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

/**
  * A hyper parameter is not within legal boundaries.
  */
final class HyperParameterOutOfBounds(val name:  String,
                                      val value: Real)
  extends IndependentOptimizerExitCode {

  override def toString
  : String = f"HyperParameterOutOfBounds[$name%s, $value%.4g]"

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[HyperParameterOutOfBounds]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: HyperParameterOutOfBounds =>
      name  == other.name &&
      value == other.value
    case _ =>
      false
  })

  override def description
  : String = f"$value%.4g is an invalid value for the '$name%s'!"

  override def indicatesConvergence
  : Boolean = false

  override def indicatesFailure
  : Boolean = true

}

object HyperParameterOutOfBounds {

  final def apply(name: String, value: Real)
  : HyperParameterOutOfBounds = new HyperParameterOutOfBounds(name, value)

}