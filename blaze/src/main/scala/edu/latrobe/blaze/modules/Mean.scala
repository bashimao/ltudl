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

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules.jvm._

/**
  *          n
  *         ---
  *       1 \
  * y_j = - /   x_ji
  *       n ---
  *          i
  *
  * d y_j    1
  * ------ = -
  * d x_ji   n
  *
  */
abstract class Mean
  extends AggregationLayer[MeanBuilder] {
}

final class MeanBuilder
  extends AggregationLayerBuilder[MeanBuilder] {

  override def repr
  : MeanBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[MeanBuilder]

  override protected def doCopy()
  : MeanBuilder = MeanBuilder()

  override def outputPlatformFor(hints: BuildHints)
  : Platform = MeanBuilder.outputPlatformFor(this, hints)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = MeanBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object MeanBuilder
  extends ModuleVariantTable[MeanBuilder] {

  register(2, Mean_JVM_Baseline_Description)

  final def apply()
  : MeanBuilder = new MeanBuilder

  final def apply(domain: TensorDomain)
  : MeanBuilder = apply().setDomain(domain)

}
