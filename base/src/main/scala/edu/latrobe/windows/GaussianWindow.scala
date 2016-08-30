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
 * Gaussian window
 *                                            2
 *                     (   n - (N - 1) / 2   )
 *                -0.5 ( ------------------- )
 *                     (        sigma        )
 * gaussian(n) = e
 *
 * Remarks:
 * Extends until infinity like normal gaussians. Hence:
 * gaussian(0) = gaussian(N-1) != 0 for n != INF
 *
 * Equations and description can be found at:
 * http://en.wikipedia.org/wiki/Window_function#Gaussian_window
 * or
 * http://au.mathworks.com/help/signal/ref/gausswin.html
 */
final class GaussianWindow(override val noWeights: Int, val sigma: Double)
  extends Window {
  require(noWeights > 0)
  require(sigma     > 0.0)

  override def toString: String = f"GaussianWindow[$noWeights%d, $sigma%.4g]"

  private val nSub1Div2: Double = 0.5 * (noWeights - 1.0)

  override def apply(index: Int): Real = {
    val x = (index - nSub1Div2) / sigma
    Real(Math.exp(-0.5 * x * x))
  }

}

object GaussianWindow {

  final def apply(noValues: Int, sigma: Double = 0.4)
  : GaussianWindow = new GaussianWindow(noValues, sigma)

}
