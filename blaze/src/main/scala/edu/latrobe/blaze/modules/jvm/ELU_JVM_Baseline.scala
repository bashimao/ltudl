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
import edu.latrobe.blaze.modules.ELUBuilder

final class ELU_JVM_Baseline(override val builder:        ELUBuilder,
                             override val inputHints:     BuildHints,
                             override val seed:           InstanceSeed,
                             override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends ELU_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(output: RealArrayTensor)
  : Unit = {
    output.transform(
      x => if (x > Real.zero) x else Real(Math.expm1(x) * alpha)
    )
  }

  override protected def doPredictInv(input: RealArrayTensor)
  : Unit = {
    val alphaInv = 1.0 / alpha
    input.transform(
      y => if (y > Real.zero) y else Real(Math.log1p(y * alphaInv))
    )
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(output: RealArrayTensor,
                                            error:  RealArrayTensor)
  : Unit = {
    error.transform(output,
      (e, y) => if (y > Real.zero) e else e * (y + alpha)
    )
  }

}

object ELU_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[ELUBuilder] {

  final override def build(builder:        ELUBuilder,
                           hints:          BuildHints,
                           seed:           InstanceSeed,
                           weightsBuilder: ValueTensorBufferBuilder)
  : ELU_JVM_Baseline = new ELU_JVM_Baseline(
    builder, hints, seed, weightsBuilder
  )

}
