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
 * Lanczos Window
 *
 * sinc(0) = 1
 *
 *           sin( pi x )
 * sinc(x) = -----------
 *              pi x
 *
 *  n = x + offset
 *  N = noValues + 2 * offset
 *
 *                  (  2 n      )
 * lanczos(x) = sinc( ----- - 1 )
 *                  ( N - 1     )
 *
 * Remarks:
 * if offset = 0 then lanczos(0) = lanczos(N-1) = 0
 *
 * A lanczos window, normalized to sum(w) = 1.
 *
 * Note that w(0) and w(N) = 0. Since all lookups by our systems use discrete
 * indices, we will discard the values from our window by inflating N by 1.
 * So n in [1, N-1]!
 *
 *                     w(n)
 * normalized_w(n) = --------
 *                   N-1
 *                   ---
 *                   \
 *                   /   w(i)
 *                   ---
 *                   i=1
 */
final class LanczosWindow(override val noWeights: Int, val offset: Double)
  extends Window {
  require(noWeights >  0)
  require(offset    >= 0.0)

  override def toString: String = f"LanczosWindow[$noWeights%d, $offset%.4g]"

  private val nSubOne: Double = {
    val N = noWeights + offset + offset
    N - 1.0
  }

  override def apply(index: Int): Real = {
    val n = index + offset
    val x = Math.PI * (n + n) / nSubOne - 1.0
    Real(Math.sin(x) / x)
  }

}

object LanczosWindow {

  final def apply(noWeights: Int, offset: Real = Real.pointFive)
  : LanczosWindow = new LanczosWindow(noWeights, offset)

}
