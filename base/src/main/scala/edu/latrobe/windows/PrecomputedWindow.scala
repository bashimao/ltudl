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
 * Default weighting window implementation that is backed by a look up table.
 */
final class PrecomputedWindow(private val weights: Array[Real])
  extends Window {

  override val noWeights: Int = weights.length

  // ---------------------------------------------------------------------------
  //    Weights related.
  // ---------------------------------------------------------------------------
  override def apply(index: Int): Real = weights(index)

}

object PrecomputedWindow {

  final def apply(weights: Array[Real])
  : PrecomputedWindow = new PrecomputedWindow(weights)

}
