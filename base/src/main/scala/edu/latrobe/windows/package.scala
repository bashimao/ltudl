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

package edu.latrobe

package object windows {

  final implicit class WindowFunctions(w: Window) {

    def cache
    : PrecomputedWindow = {
      val weights = new Array[Real](w.noWeights)
      w.foreachWeightPair((i, w) => weights(i) = w)
      PrecomputedWindow(weights)
    }

    def normalize()
    : PrecomputedWindow = {
      val sum     = w.sum
      val weights = new Array[Real](w.noWeights)
      w.foreachWeightPair((i, w) => weights(i) = w / sum)
      PrecomputedWindow(weights)
    }

    /**
      * Creates a multidimensional window. Use this for 2, 3 and N
      * dimensional windows. this" becomes the lower dimensions, "other"
      * becomes the higher dimensions.
      */
    def *(other: Window)
    : PrecomputedWindow = {
      val n0      = w.noWeights
      val n1      = other.noWeights
      val weights = new Array[Real](n0 * n1)
      var i       = 0
      while (i < weights.length) {
        weights(i) = w(i % n0) * other(i / n0)
        i += 1
      }
      PrecomputedWindow(weights)
    }

  }

}
