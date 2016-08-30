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
import edu.latrobe.blaze.modules.PReLUBuilder

final class PReLU_JVM_Baseline(override val builder:        PReLUBuilder,
                               override val inputHints:     BuildHints,
                               override val seed:           InstanceSeed,
                               override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends PReLU_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(output: RealArrayTensor)
  : Unit = {
    val out = output.values
    val w   = pReLU.values
    var off = 0
    while (off < out.length) {
      ArrayEx.transform(
        out, off, 1,
        w,   0,   1,
        w.length
      )((x, w) => if (x >= 0) x else x * w)
      off += w.length
    }
    assume(off == out.length)
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDerivePReLUGradients(input: RealArrayTensor,
                                                error: RealArrayTensor,
                                                sink:  RealArrayTensor)
  : Unit = {
    val inp = input.values
    val err = error.values
    val dst = sink.values
    var off = 0
    while (off < err.length) {
      ArrayEx.transform(
        dst, 0,   1,
        inp, off, 1,
        err, off, 1,
        dst.length
      )((d, x, e) => if (x >= 0) d else d + e * x)
      off += dst.length
    }
    assume(off == err.length)
  }

  override protected def doDeriveInputError(input: RealArrayTensor,
                                            error: RealArrayTensor)
  : Unit = {
    val inp = input.values
    val err = error.values
    val w   = pReLU.values
    var off = 0
    while (off < err.length) {
      ArrayEx.transform(
        err, off, 1,
        inp, off, 1,
        w,   0,   1,
        w.length
      )((e, x, w) => if (x >= 0) e else e * w)
      off += w.length
    }
    assume(off == err.length)
  }

}

object PReLU_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[PReLUBuilder] {

  override def build(builder:        PReLUBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : PReLU_JVM_Baseline = new PReLU_JVM_Baseline(
    builder, hints, seed, weightsBuilder
  )

}
