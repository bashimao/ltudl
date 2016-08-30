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

package edu.latrobe.blaze.initializers

import edu.latrobe._

object InitializerGain {

  final val forLinear
  : Real = Real.one

  final val forSigmoid
  : Real = Real.one

  final val forReLU
  : Real = Real(Math.sqrt(2.0))

  final def forLeakyReLU(threshold: Real)
  : Real = Real(Math.sqrt(2.0 / (1 + threshold * threshold)))

}
