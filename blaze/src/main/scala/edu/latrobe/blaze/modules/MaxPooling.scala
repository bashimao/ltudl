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
 * Performs max pooling over several neurons in several maps.
 */
abstract class MaxPooling
  extends PoolingLayerEx[MaxPoolingBuilder] {
}

final class MaxPoolingBuilder
  extends PoolingLayerExBuilder[MaxPoolingBuilder] {

  override def repr
  : MaxPoolingBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[MaxPoolingBuilder]

  override protected def doCopy()
  : MaxPoolingBuilder = MaxPoolingBuilder()


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(hints: BuildHints)
  : Platform = MaxPoolingBuilder.outputPlatformFor(this, hints)

  // TODO: Check for unsupported combinations of settings.
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = MaxPoolingBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object MaxPoolingBuilder
  extends ModuleVariantTable[MaxPoolingBuilder] {

  register(2, MaxPooling_JVM_Baseline_Description)

  final def apply()
  : MaxPoolingBuilder = new MaxPoolingBuilder

  final def apply(kernel: Kernel)
  : MaxPoolingBuilder = apply().setKernel(kernel)

}
