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

package edu.latrobe.blaze.modules.jvm

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._

final class AddUniformNoise_JVM_Baseline(override val builder:             AddUniformNoiseBuilder,
                                         override val inputHints:          BuildHints,
                                         override val seed:                InstanceSeed,
                                         override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends AddUniformNoise_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(output: RealArrayTensor)
  : Unit = {
    ArrayEx.transform(
      output.values
    )(_ + uniform.sample())
  }

}

object AddUniformNoise_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[AddUniformNoiseBuilder] {

  override def build(builder:             AddUniformNoiseBuilder,
                     hints:               BuildHints,
                     seed:                InstanceSeed,
                     weightBufferBuilder: ValueTensorBufferBuilder)
  : Module = new AddUniformNoise_JVM_Baseline(
    builder,
    hints,
    seed,
    weightBufferBuilder
  )

}
