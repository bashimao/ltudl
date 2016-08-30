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
import scala.collection._

final class SwitchPlatform_JVM_Baseline(override val builder:             SwitchPlatformBuilder,
                                        override val inputHints:          BuildHints,
                                        override val seed:                InstanceSeed,
                                        override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends SwitchPlatform
    with MapLayer_JVM[SwitchPlatformBuilder] {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input: Tensor)
  : Tensor = input.asOrToRealArrayTensor

}

object SwitchPlatform_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[SwitchPlatformBuilder] {

  override protected def doScore(builder:   SwitchPlatformBuilder,
                                 hints:     BuildHints,
                                 scorePrev: Int,
                                 reasons:   mutable.ArrayBuilder[String])
  : Int = {
    var score = super.doScore(builder, hints, scorePrev, reasons)
    builder.preferredPlatform.foreach({
      case JVM =>
        score |= (1 << 30)
      case _ =>
    })
    score
  }

  override def build(builder:        SwitchPlatformBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : SwitchPlatform_JVM_Baseline = new SwitchPlatform_JVM_Baseline(
    builder, hints, seed, weightsBuilder
  )

}
