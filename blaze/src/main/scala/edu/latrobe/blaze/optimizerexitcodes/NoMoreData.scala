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
  * No more data available in data stream. Does not happen with infinite streams.
  */
final class NoMoreData
  extends IndependentOptimizerExitCode {

  override def toString
  : String = "NoMoreData"

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[NoMoreData]

  override def description
  : String = {
    "The batch pool that feeds the optimization process has been depleted!"
  }

  override def indicatesConvergence
  : Boolean = false

  override def indicatesFailure
  : Boolean = false

}

object NoMoreData {

  final def apply()
  : NoMoreData = new NoMoreData

}
