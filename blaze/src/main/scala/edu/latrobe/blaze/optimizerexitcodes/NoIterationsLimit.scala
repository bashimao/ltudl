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
  * For example: Number of iterations exceeds some fancy limit (pow(2,64) - 1).
  * This should never happen!
  */
final class NoIterationsLimit
  extends IndependentOptimizerExitCode {

  override def toString
  : String = "NoIterationsLimit"

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[NoIterationsLimit]

  override def description
  : String = "Number of iterations exceeds hardware limit!"

  override def indicatesConvergence
  : Boolean = false

  override def indicatesFailure
  : Boolean = false

}

object NoIterationsLimit {

  final def apply()
  : NoIterationsLimit = new NoIterationsLimit

}
