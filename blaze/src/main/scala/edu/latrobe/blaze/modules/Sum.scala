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

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules.jvm._

/**
  *        n
  *       ---
  *       \
  * y_j = /   x_ji
  *       ---
  *        i
  *
  * d y_j
  * ------ = 1
  * d x_ji
  *
  */
abstract class Sum
  extends AggregationLayer[SumBuilder] {
}

final class SumBuilder
  extends AggregationLayerBuilder[SumBuilder] {

  override def repr
  : SumBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[SumBuilder]

  override protected def doCopy()
  : SumBuilder = SumBuilder()

  override def outputPlatformFor(hints: BuildHints)
  : Platform = SumBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = SumBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object SumBuilder
  extends ModuleVariantTable[SumBuilder] {

  register(2, Sum_JVM_Baseline_Description)

  final def apply()
  : SumBuilder = new SumBuilder

  final def apply(domain: TensorDomain)
  : SumBuilder = apply().setDomain(domain)

}
