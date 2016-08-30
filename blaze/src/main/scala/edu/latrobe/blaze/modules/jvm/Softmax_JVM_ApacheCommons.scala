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
import edu.latrobe.blaze.modules.SoftmaxBuilder
import org.apache.commons.math3.util._

final class Softmax_JVM_ApacheCommons(override val builder:        SoftmaxBuilder,
                                      override val inputHints:     BuildHints,
                                      override val seed:           InstanceSeed,
                                      override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Softmax_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(output: RealArrayTensor)
  : Unit = {
    val out = output.values

    output.foreachSample((off, length) => {
      // Subtract max to make things numerically a little bit more stable.
      val max = DoubleEx(ArrayEx.max(
        out, off, 1,
        length
      ))

      var sum: Double = Real.minQuotient1
      ArrayEx.transform(
        out, off, 1,
        length
      )(x => {
        val tmp = FastMath.exp(x - max)
        sum += tmp
        Real(tmp)
      })

      // Scale results by sum to squash them between 0 and 1.
      ArrayEx.multiply(
        out, off, 1,
        Real(1.0 / sum),
        length
      )
    })
  }

}

object Softmax_JVM_ApacheCommons_Description
  extends ModuleVariant_JVM_Description[SoftmaxBuilder] {

  override def build(builder:        SoftmaxBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Softmax_JVM_ApacheCommons = new Softmax_JVM_ApacheCommons(
    builder, hints, seed, weightsBuilder
  )

}
