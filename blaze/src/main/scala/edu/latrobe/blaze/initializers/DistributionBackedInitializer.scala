/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2015 Matthias Langer (t3l@threelights.de)
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

import breeze.stats.distributions._
import edu.latrobe._
import edu.latrobe.blaze._

/**
  * Simple initializers are directly backed by a distribution from which they
  * sample values to fill the buffers.
  */
abstract class DistributionBackedInitializer[TBuilder <: DistributionBackedInitializerBuilder[_]]
  extends IndependentInitializer[TBuilder] {

  /**
    * Should override this with val.
    */
  def distribution
  : Distribution[Real]

  final override def apply(module:        Module,
                           reference:     LabeledBufferReference,
                           weights:       ValueTensor,
                           inputFanSize:  Int,
                           outputFanSize: Int)
  : Unit = weights.fill(distribution.sample, threadSafe = false)

}

abstract class DistributionBackedInitializerBuilder[TThis <: DistributionBackedInitializerBuilder[_]]
  extends IndependentInitializerBuilder[TThis] {
}
