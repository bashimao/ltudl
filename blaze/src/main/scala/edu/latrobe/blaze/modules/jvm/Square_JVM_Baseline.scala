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

final class Square_JVM_Baseline(override val builder:        SquareBuilder,
                                override val inputHints:     BuildHints,
                                override val seed:           InstanceSeed,
                                override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Square_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(output: RealArrayTensor)
  : Unit = {
    output.transform(
      x => x * x
    )
  }

  override protected def doPredictInv(input: RealArrayTensor)
  : Unit = {
    input.transform(
      y => Real(Math.sqrt(DoubleEx(y)))
    )
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(input: RealArrayTensor,
                                            error: RealArrayTensor)
  : Unit = {
    error.transform(input,
      (e, x) => e * (x + x)
    )
  }

}

object Square_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[SquareBuilder] {

  override def build(builder:        SquareBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Square_JVM_Baseline = new Square_JVM_Baseline(
    builder, hints, seed, weightsBuilder
  )

}
