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

package edu.latrobe.windows

import edu.latrobe._

/**
  * Welch Window (=Parabola/Quadratic Window)
  *
  * https://en.wikipedia.org/wiki/Window_function#Welch_window
  *
  *  n = x + offset
  *  N = noValues + 2 * offset
  *
  *                             2
  *                (     N - 1 )
  *                ( n - ----- )
  *                (       2   )
  * welch(x) = 1 - ( --------- )
  *                (   N - 1   )
  *                (   -----   )
  *                (     2     )
  *
  * Remarks:
  * if offset = 0 then welch(0) = welch(N-1) = 0
  *
  * Normalized! Also see description of lanczos window!
  */
final class WelchWindow(override val noWeights: Int, val offset: Double)
  extends Window {
  require(noWeights >  0)
  require(offset    >= 0.0)

  override def toString: String = f"WelchWindow[$noWeights%d, $offset%.4g]"

  private val nSub1Div2: Double = {
    val N = noWeights + offset + offset
    0.5 * (N - 1.0)
  }

  override def apply(index: Int): Real = {
    val n = index + offset
    val x = (n - nSub1Div2) / nSub1Div2
    Real(1.0 - x * x)
  }

}

object WelchWindow {

  final def apply(noWeights: Int, offset: Real = Real.pointFive)
  : WelchWindow = new WelchWindow(noWeights, offset)

}
