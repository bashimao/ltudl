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

final class MultiplyFilter_JVM_Baseline(override val builder:        MultiplyFilterBuilder,
                                        override val inputHints:     BuildHints,
                                        override val seed:           InstanceSeed,
                                        override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends MultiplyFilter_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredictPerValue(output: RealArrayTensor)
  : Unit = {
    // input:  RGB RGB RGB | RGB RGB RGB
    // filter: RGB RGB RGB | RGB RGB RGB
    output :*= filter
  }

  override protected def doPredictPerUnit(output: RealArrayTensor)
  : Unit = {
    // input:  RGB RGB RGB | RGB RGB RGB
    // filter: RGB RGB RGB
    output.foreachSample((off, length) => {
      ArrayEx.multiply(
        output.values, off, 1,
        filter.values, 0,   1,
        length
      )
    })
  }

  override protected def doPredictPerChannel(output: RealArrayTensor)
  : Unit = {
    // input:  RGB RGB RGB | RGB RGB RGB
    // filter: RGB
    output.foreachChannel((off, stride, length) => {
      ArrayEx.multiply(
        output.values, off, stride,
        filter.values(off),
        length
      )
    })
    /*
    // TODO: Parallelize!
    val out = output.values
    val w   = filter.values
    var off = 0
    while (off < out.length) {
      ArrayEx.multiply(
        out, off, 1,
        w,   0,   1,
        w.length
      )
      off += w.length
    }
    assume(off == out.length)
    */
  }

  override protected def doPredictPerSample(output: RealArrayTensor)
  : Unit = {
    // input:  RGB RGB RGB | RGB RGB RGB
    // filter: R           | R
    output.foreachSamplePair((i, off, length) => {
      ArrayEx.multiply(
        output.values, off, 1,
        filter.values(i),
        length
      )
    })
  }

  override protected def doPredictPerBatch(output: RealArrayTensor)
  : Unit = {
    // input: RGB RGB RGB | RGB RGB RGB
    // bias:  R
    output *= filter.values(0)
  }


  override protected def doPredictInvPerValue(input: RealArrayTensor)
  : Unit = {
    // input:  RGB RGB RGB | RGB RGB RGB
    // filter: RGB RGB RGB | RGB RGB RGB
    input :/= filter
  }

  override protected def doPredictInvPerUnit(input: RealArrayTensor)
  : Unit = {
    // input:  RGB RGB RGB | RGB RGB RGB
    // filter: RGB RGB RGB
    input.foreachUnit((off, stride, length) => {
      ArrayEx.multiply(
        input.values, off, stride,
        Real.one / filter.values(off),
        length
      )
    })
  }

  override protected def doPredictInvPerChannel(input: RealArrayTensor)
  : Unit = {
    // input:  RGB RGB RGB | RGB RGB RGB
    // filter: RGB
    input.foreachChannel((off, stride, length) => {
      ArrayEx.multiply(
        input.values, off, stride,
        Real.one / filter.values(off),
        length
      )
    })
    /*
    // TODO: Parallelize!
    val inp = input.values
    val w   = filter.values
    var off = 0
    while (off < inp.length) {
      ArrayEx.divide(
        inp, off, 1,
        w,   0,   0,
        w.length
      )
      off += w.length
    }
    assume(off == inp.length)
    */
  }

  override protected def doPredictInvPerSample(input: RealArrayTensor)
  : Unit = {
    // input:  RGB RGB RGB | RGB RGB RGB
    // filter: R           | R
    input.foreachSamplePair((i, off, length) => {
      ArrayEx.multiply(
        input.values,  off, 1,
        Real.one / filter.values(i),
        length
      )
    })
  }

  override protected def doPredictInvPerBatch(input: RealArrayTensor)
  : Unit = {
    // input:  RGB RGB RGB | RGB RGB RGB
    // filter: R
    input *= Real.one / filter.values(0)
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveFilterGradientsPerValue(input: RealArrayTensor,
                                                         error: RealArrayTensor,
                                                         sink:  RealArrayTensor)
  : Unit = {
    // input: RGB RGB RGB | RGB RGB RGB
    // error: RGB RGB RGB | RGB RGB RGB
    // sink:  RGB RGB RGB | RGB RGB RGB
    sink.add(
      error,
      input
    )
  }

  override protected def doDeriveFilterGradientsPerUnit(input: RealArrayTensor,
                                                        error: RealArrayTensor,
                                                        sink:  RealArrayTensor)
  : Unit = {
    // input: RGB RGB RGB | RGB RGB RGB
    // error: RGB RGB RGB | RGB RGB RGB
    // sink:  RGB RGB RGB
    error.foreachUnit((off, stride, length) => {
      val tmp = ArrayEx.dot(
        error.values, off, stride,
        input.values, off, stride,
        length
      )
      sink.values(off) += tmp
    })
  }

  override protected def doDeriveFilterGradientsPerChannel(input: RealArrayTensor,
                                                           error: RealArrayTensor,
                                                           sink:  RealArrayTensor)
  : Unit = {
    // input: RGB RGB RGB | RGB RGB RGB
    // error: RGB RGB RGB | RGB RGB RGB
    // sink:  RGB
    error.foreachChannel((off, stride, length) => {
      val tmp = ArrayEx.dot(
        error.values, off, stride,
        input.values, off, stride,
        length
      )
      sink.values(off) += tmp
    })
    /*
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
      )(_ + _ * _)
      off += dst.length
    }
    assume(off == err.length)
    */
  }

  override protected def doDeriveFilterGradientsPerSample(input: RealArrayTensor,
                                                          error: RealArrayTensor,
                                                          sink:  RealArrayTensor)
  : Unit = {
    // input: RGB RGB RGB | RGB RGB RGB
    // error: RGB RGB RGB | RGB RGB RGB
    // sink:  R           | R
    error.foreachSamplePair((i, off, length) => {
      val tmp = ArrayEx.dot(
        error.values, off, 1,
        input.values, off, 1,
        length
      )
      sink.values(i) += tmp
    })
  }

  override protected def doDeriveFilterGradientsPerBatch(input: RealArrayTensor,
                                                         error: RealArrayTensor,
                                                         sink:  RealArrayTensor)
  : Unit = {
    // input: RGB RGB RGB | RGB RGB RGB
    // error: RGB RGB RGB | RGB RGB RGB
    // sink:  R
    sink.values(0) += error.dot(input)
  }

  override protected def doDeriveInputErrorPerValue(error: RealArrayTensor)
  : Unit = {
    // error:  RGB RGB RGB | RGB RGB RGB
    // filter: RGB RGB RGB | RGB RGB RGB
    error :*= filter
  }

  override protected def doDeriveInputErrorPerUnit(error: RealArrayTensor)
  : Unit = {
    // error:  RGB RGB RGB | RGB RGB RGB
    // filter: RGB RGB RGB
    error.foreachSample((off, length) => {
      ArrayEx.multiply(
        error.values,  off, 1,
        filter.values, 0,   1,
        length
      )
    })
  }

  override protected def doDeriveInputErrorPerChannel(error: RealArrayTensor)
  : Unit = {
    // error:  RGB RGB RGB | RGB RGB RGB
    // filter: RGB
    error.foreachChannel((off, stride, length) => {
      ArrayEx.multiply(
        error.values, off, stride,
        filter.values(off),
        length
      )
    })
    /*
    // TODO: Parallelize!
    val err = error.values
    val w   = filter.values
    var off = 0
    while (off < err.length) {
      ArrayEx.multiply(
        err, off, 1,
        w,   0,   1,
        w.length
      )
      off += w.length
    }
    assume(off == err.length)
    */
  }

  override protected def doDeriveInputErrorPerSample(error: RealArrayTensor)
  : Unit = {
    // error:  RGB RGB RGB | RGB RGB RGB
    // filter: R           | R
    error.foreachSamplePair((i, off, length) => {
      ArrayEx.multiply(
        error.values, off, 1,
        filter.values(i),
        length
      )
    })
  }

  override protected def doDeriveInputErrorPerBatch(error: RealArrayTensor)
  : Unit = {
    // error:  RGB RGB RGB | RGB RGB RGB
    // filter: R
    error *= filter.values(0)
  }

}

object ImmediateFilter_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[MultiplyFilterBuilder] {

  override def build(builder:        MultiplyFilterBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : MultiplyFilter_JVM_Baseline = new MultiplyFilter_JVM_Baseline(
    builder, hints, seed, weightsBuilder
  )

}
