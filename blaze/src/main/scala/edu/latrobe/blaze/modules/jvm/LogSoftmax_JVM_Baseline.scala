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
import edu.latrobe.blaze.modules.LogSoftmaxBuilder

final class LogSoftmax_JVM_Baseline(override val builder:        LogSoftmaxBuilder,
                                    override val inputHints:     BuildHints,
                                    override val seed:           InstanceSeed,
                                    override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends LogSoftmax_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(output: RealArrayTensor)
  : Unit = {
    val out = output.values

    output.foreachSample((off, length) => {
      // Compute logSum, but subtract max to make things numerically a little bit more stable.
      val max = DoubleEx(ArrayEx.max(
        out, off, 1,
        length
      ))

      var logSum = 0.0
      ArrayEx.foreach(
        out, off, 1,
        length
      )(x => logSum += Math.exp(x - max))
      logSum = Math.max(logSum, DoubleEx.epsilon)
      logSum = Math.log(logSum) + max

      // Subtract logSum.
      ArrayEx.add(
        out, off, 1,
        Real(-logSum),
        length
      )
    })
  }

  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(output: RealArrayTensor,
                                            error:  RealArrayTensor)
  : Unit = {
    val out = output.values
    val err = error.values

    error.foreachSample((off, length) => {
      val sum = ArrayEx.sum(
        err, off, 1,
        length
      )
      ArrayEx.transform(
        err, off, 1,
        out, off, 1,
        length
      )((e, y) => Real(e - Math.exp(y) * sum))
    })
  }

}

object LogSoftmax_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[LogSoftmaxBuilder] {

  override def build(builder:        LogSoftmaxBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : LogSoftmax_JVM_Baseline = new LogSoftmax_JVM_Baseline(
    builder, hints, seed, weightsBuilder
  )

}

final case class LogSoftmax_JVM_Baseline_Context(logSums: Array[Double])
  extends PredictContext {
}