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

final class MultiplyValues_JVM_Baseline(override val builder:        MultiplyValuesBuilder,
                                        override val inputHints:     BuildHints,
                                        override val seed:           InstanceSeed,
                                        override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends MultiplyValues_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredictPerValue(output: RealArrayTensor)
  : Unit = {
    // input:  RGB RGB RGB | RGB RGB RGB
    // values: RGB RGB RGB | RGB RGB RGB
    output.foreachSample((off, length) => {
      ArrayEx.multiply(
        output.values, off, 1,
        values,        off, 1,
        length
      )
    })
  }

  override protected def doPredictPerUnit(output: RealArrayTensor)
  : Unit = {
    // input:  RGB RGB RGB | RGB RGB RGB
    // values: RGB RGB RGB
    output.foreachSample((off, length) => {
      ArrayEx.multiply(
        output.values, off, 1,
        values,        0,   1,
        length
      )
    })
  }

  override protected def doPredictPerChannel(output: RealArrayTensor)
  : Unit = {
    // input:  RGB RGB RGB | RGB RGB RGB
    // values: RGB
    output.foreachChannel((off, stride, length) => {
      ArrayEx.multiply(
        output.values, off, stride,
        values(off),
        length
      )
    })
    /*
    val sampleSize = output.layout.size.noValues
    output.foreachSample(offset0 => {
      var off0 = offset0
      val end0 = offset0 + sampleSize
      while (off0 < end0) {
        ArrayEx.multiply(
          output.values, off0, 1,
          values,        0,    1,
          values.length
        )
        off0 += values.length
      }
    })
    */
  }

  override protected def doPredictPerSample(output: RealArrayTensor)
  : Unit = {
    // input:  RGB RGB RGB | RGB RGB RGB
    // values: R           | R
    output.foreachSamplePair((i, off, length) => {
      ArrayEx.multiply(
        output.values, off, 1,
        values(i),
        length
      )
    })
  }

  override protected def doPredictPerBatch(output: RealArrayTensor)
  : Unit = {
    // input:  RGB RGB RGB | RGB RGB RGB
    // values: R
    output *= values(0)
  }

  override protected def doPredictInvPerValue(input: RealArrayTensor)
  : Unit = {
    // output: RGB RGB RGB | RGB RGB RGB
    // values: RGB RGB RGB | RGB RGB RGB
    input.foreachSample((off, length) => {
      ArrayEx.divide(
        input.values, off, 1,
        values,       off, 1,
        length
      )
    })
  }

  override protected def doPredictInvPerUnit(input: RealArrayTensor)
  : Unit = {
    // output: RGB RGB RGB | RGB RGB RGB
    // values: RGB RGB RGB
    input.foreachSample((off, length) => {
      ArrayEx.divide(
        input.values, off, 1,
        values,       0,   1,
        length
      )
    })
  }

  override protected def doPredictInvPerChannel(input: RealArrayTensor)
  : Unit = {
    // output: RGB RGB RGB | RGB RGB RGB
    // values: RGB
    input.foreachChannel((off, stride, length) => {
      ArrayEx.multiply(
        input.values, off, stride,
        Real.one / values(off),
        length
      )
    })
    /*
    val sampleSize = input.layout.size.noValues
    input.foreachSample(offset0 => {
      var off0 = offset0
      val end0 = offset0 + sampleSize
      while (off0 < end0) {
        ArrayEx.divide(
          input.values, off0, 1,
          values, 0, 1,
          values.length
        )
        off0 += values.length
      }
    })
    */
  }

  override protected def doPredictInvPerSample(input: RealArrayTensor)
  : Unit = {
    // output: RGB RGB RGB | RGB RGB RGB
    // values: R           | R
    input.foreachSamplePair((i, off, length) => {
      ArrayEx.multiply(
        input.values, off, 1,
        Real.one / values(i),
        length
      )
    })
  }

  override protected def doPredictInvPerBatch(input: RealArrayTensor)
  : Unit = {
    // output: RGB RGB RGB | RGB RGB RGB
    // bias:   R
    input *= Real.one / values(0)
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputErrorPerValue(error: RealArrayTensor)
  : Unit = {
    // error:  RGB RGB RGB | RGB RGB RGB
    // values: RGB RGB RGB | RGB RGB RGB
    error.foreachChannel((off, stride, length) => {
      ArrayEx.multiply(
        error.values, off, stride,
        values,       off, stride,
        length
      )
    })
  }

  override protected def doDeriveInputErrorPerUnit(error: RealArrayTensor)
  : Unit = {
    // error:  RGB RGB RGB | RGB RGB RGB
    // values: RGB RGB RGB
    error.foreachSample((off, length) => {
      ArrayEx.multiply(
        error.values, off, 1,
        values,       0,   1,
        length
      )
    })
  }

  override protected def doDeriveInputErrorPerChannel(error: RealArrayTensor)
  : Unit = {
    // error:  RGB RGB RGB | RGB RGB RGB
    // values: RGB
    error.foreachChannel((off, stride, length) => {
      ArrayEx.multiply(
        error.values, off, stride,
        values(off),
        length
      )
    })
    /*
    val sampleSize = error.layout.size.noValues
    error.foreachSample(offset0 => {
      var off0 = offset0
      val end0 = offset0 + sampleSize
      while (off0 < end0) {
        ArrayEx.multiply(
          error.values, off0, 1,
          values,       0,    1,
          values.length
        )
        off0 += values.length
      }
    })
    */
  }

  override protected def doDeriveInputErrorPerSample(error: RealArrayTensor)
  : Unit = {
    // error:  RGB RGB RGB | RGB RGB RGB
    // values: R           | R
    error.foreachSamplePair((i, off, length) => {
      ArrayEx.multiply(
        error.values, off, 1,
        values(i),
        length
      )
    })
  }

  override protected def doDeriveInputErrorPerBatch(error: RealArrayTensor)
  : Unit = {
    // error:  RGB RGB RGB | RGB RGB RGB
    // values: R
    error *= values(0)
  }

}

object MultiplyValues_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[MultiplyValuesBuilder] {

  override def build(builder:        MultiplyValuesBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : MultiplyValues_JVM_Baseline = new MultiplyValues_JVM_Baseline(
    builder, hints, seed, weightsBuilder
  )

}