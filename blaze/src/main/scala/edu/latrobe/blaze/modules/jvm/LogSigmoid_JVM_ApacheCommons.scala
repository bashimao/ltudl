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
import edu.latrobe.blaze.modules.LogSigmoidBuilder
import org.apache.commons.math3.util._

final class LogSigmoid_JVM_ApacheCommons(override val builder:        LogSigmoidBuilder,
                                         override val inputHints:     BuildHints,
                                         override val seed:           InstanceSeed,
                                         override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends LogSigmoid_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(output: RealArrayTensor)
  : Unit = output.transform(
    x => Real(-FastMath.log1p(FastMath.exp(-x)))
  )

  override protected def doPredictInv(input: RealArrayTensor)
  : Unit = input.transform(
    y => Real(-FastMath.log(FastMath.expm1(-y)))
  )


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(input: RealArrayTensor,
                                            error: RealArrayTensor)
  : Unit = error.transform(input, (dy, x) => {
    val tmp = FastMath.exp(-x)
    Real(dy * (tmp / (tmp + 1.0)))
  })

}

object LogSigmoid_JVM_ApacheCommons_Description
  extends ModuleVariant_JVM_Description[LogSigmoidBuilder] {

  override def build(builder:        LogSigmoidBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : LogSigmoid_JVM_ApacheCommons = new LogSigmoid_JVM_ApacheCommons(
    builder, hints, seed, weightsBuilder
  )

}
