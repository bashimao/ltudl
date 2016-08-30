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
 * Has uniform weights across the entire window.
 *
 * Rectangular window function:
 * http://en.wikipedia.org/wiki/Window_function#Rectangular_window
 */
final class RectangularWindow(override val noWeights: Int)
  extends Window {
  require(noWeights > 0)

  override def toString: String = s"RectangularWindow[$noWeights]"

  private val noWeightsInv = Real.one / noWeights

  override def apply(index: Int): Real = noWeightsInv

}

object RectangularWindow {

  final def apply(noWeights: Int)
  : RectangularWindow = new RectangularWindow(noWeights)

  final def apply(dims: Tuple1[Int]): RectangularWindow = apply(dims._1)

  final def apply(dims: (Int, Int)): RectangularWindow = {
    require(dims._1 >= 0)
    require(dims._2 >= 0)
    apply(dims._1 * dims._2)
  }

  final def apply(dims: (Int, Int, Int)): RectangularWindow = {
    require(dims._1 >= 0)
    require(dims._2 >= 0)
    require(dims._3 >= 0)
    apply(dims._1 * dims._2 * dims._3)
  }

  final def apply(dims: (Int, Int, Int, Int)): RectangularWindow = {
    require(dims._1 >= 0)
    require(dims._2 >= 0)
    require(dims._3 >= 0)
    require(dims._4 >= 0)
    apply(dims._1 * dims._2 * dims._3 * dims._4)
  }

}
