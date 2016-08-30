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
import edu.latrobe.blaze.modules.AddBiasBuilder

final class AddBias_JVM_Baseline(override val builder:        AddBiasBuilder,
                                 override val inputHints:     BuildHints,
                                 override val seed:           InstanceSeed,
                                 override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends AddBias_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredictPerValue(output: RealArrayTensor)
  : Unit = {
    // input:  RGB RGB RGB | RGB RGB RGB
    // values: RGB RGB RGB | RGB RGB RGB
    output += bias
  }

  override protected def doPredictPerUnit(output: RealArrayTensor)
  : Unit = {
    // input: RGB RGB RGB | RGB RGB RGB
    // bias:  RGB RGB RGB
    output.foreachSample((off, length) => {
      ArrayEx.add(
        output.values, off, 1,
        bias.values,   0,   1,
        length
      )
    })
  }

  override protected def doPredictPerChannel(output: RealArrayTensor)
  : Unit = {
    // input: RGB RGB RGB | RGB RGB RGB
    // bias:  RGB
    output.foreachChannel((off, stride, length) => {
      ArrayEx.add(
        output.values, off, stride,
        bias.values(off),
        length
      )
    })
    /*
    // TODO: Make parallel!
    val out = output.values
    val b   = bias.values
    var off = 0
    while (off < out.length) {
      ArrayEx.add(
        out, off, 1,
        b,   0,   1,
        b.length
      )
      off += b.length
    }
    assume(off == out.length)
    */
  }

  override protected def doPredictPerSample(output: RealArrayTensor)
  : Unit = {
    // input: RGB RGB RGB | RGB RGB RGB
    // bias:  R           | R
    output.foreachSamplePair((i, off, length) => {
      ArrayEx.add(
        output.values, off, 1,
        bias.values(i),
        length
      )
    })
  }

  override protected def doPredictPerBatch(output: RealArrayTensor)
  : Unit = {
    // input: RGB RGB RGB | RGB RGB RGB
    // bias:  R
    output += bias.values(0)
  }

  override protected def doPredictInvPerValue(input: RealArrayTensor)
  : Unit = {
    // input: RGB RGB RGB | RGB RGB RGB
    // bias:  RGB RGB RGB | RGB RGB RGB
    input -= bias
  }

  override protected def doPredictInvPerUnit(input: RealArrayTensor)
  : Unit = {
    // input: RGB RGB RGB | RGB RGB RGB
    // bias:  RGB RGB RGB
    input.foreachSample((off, length) => {
      ArrayEx.subtract(
        input.values, off, 1,
        bias.values,  0,   1,
        length
      )
    })
  }

  override protected def doPredictInvPerChannel(input: RealArrayTensor)
  : Unit = {
    // input: RGB RGB RGB | RGB RGB RGB
    // bias:  RGB
    input.foreachChannel((off, stride, length) => {
      ArrayEx.add(
        input.values, off, stride,
        -bias.values(off),
        length
      )
    })
    /*
    // TODO: Make parallel!
    val inp = input.values
    val b   = bias.values
    var off = 0
    while (off < inp.length) {
      ArrayEx.subtract(
        inp, off, 1,
        b,   0,   1,
        b.length
      )
      off += b.length
    }
    assume(off == inp.length)
    */
  }

  override protected def doPredictInvPerSample(input: RealArrayTensor)
  : Unit = {
    // input: RGB RGB RGB | RGB RGB RGB
    // bias:  R           | R
    input.foreachSamplePair((i, off, length) => {
      ArrayEx.add(
        input.values, off, 1,
        -bias.values(i),
        length
      )
    })
  }

  override protected def doPredictInvPerBatch(input: RealArrayTensor)
  : Unit = {
    // input: RGB RGB RGB | RGB RGB RGB
    // bias:  R
    input += -bias.values(0)
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveWeightGradientsPerValue(error: RealArrayTensor,
                                                         sink:  RealArrayTensor)
  : Unit = {
    // error: RGB RGB RGB | RGB RGB RGB
    // sink:  RGB RGB RGB | RGB RGB RGB
    sink += error
  }

  override protected def doDeriveWeightGradientsPerUnit(error: RealArrayTensor,
                                                        sink:  RealArrayTensor)
  : Unit = {
    // error: RGB RGB RGB | RGB RGB RGB
    // sink:  RGB RGB RGB
    error.foreachUnit((off, stride, length) => {
      val tmp = ArrayEx.sum(
        error.values, off, stride,
        length
      )
      sink.values(off) += tmp
    })
  }

  override protected def doDeriveWeightGradientsPerChannel(error: RealArrayTensor,
                                                           sink:  RealArrayTensor)
  : Unit = {
    // input: RGB RGB RGB | RGB RGB RGB
    // bias:  RGB
    error.foreachChannel((off, stride, length) => {
      val tmp = ArrayEx.sum(
        error.values, off, stride,
        length
      )
      sink.values(off) += tmp
    })
    /*
    val err = error.values
    val dst = sink.values
    var off = 0
    while (off < err.length) {
      ArrayEx.add(
        dst, 0,   1,
        err, off, 1,
        dst.length
      )
      off += dst.length
    }
    assume(off == err.length)
    */
  }

  override protected def doDeriveWeightGradientsPerSample(error: RealArrayTensor,
                                                          sink:  RealArrayTensor)
  : Unit = {
    // input: RGB RGB RGB | RGB RGB RGB
    // sink:  R           | R
    error.foreachSamplePair((i, off, length) => {
      val tmp = ArrayEx.sum(
        error.values, off, 1,
        length
      )
      sink.values(i) += tmp
    })
  }

  override protected def doDeriveWeightGradientsPerBatch(error: RealArrayTensor,
                                                         sink:  RealArrayTensor)
  : Unit = {
    // error: RGB RGB RGB | RGB RGB RGB
    // sink:  R
    sink.values(0) += error.sum
  }

}

object AddBias_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[AddBiasBuilder] {

  override def build(builder:        AddBiasBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : AddBias_JVM_Baseline = new AddBias_JVM_Baseline(
    builder, hints, seed, weightsBuilder
  )

}
