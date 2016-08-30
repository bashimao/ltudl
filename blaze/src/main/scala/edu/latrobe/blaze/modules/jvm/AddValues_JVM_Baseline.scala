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

final class AddValues_JVM_Baseline(override val builder:        AddValuesBuilder,
                                   override val inputHints:     BuildHints,
                                   override val seed:           InstanceSeed,
                                   override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends AddValues_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredictPerValue(output: RealArrayTensor)
  : Unit = {
    // input:  RGB RGB RGB | RGB RGB RGB
    // values: RGB RGB RGB | RGB RGB RGB
    output.foreachSample((off, length) => {
      ArrayEx.add(
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
      ArrayEx.add(
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
      ArrayEx.add(
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
        ArrayEx.add(
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
      ArrayEx.add(
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
    output += values(0)
  }

  override protected def doPredictInvPerValue(input: RealArrayTensor)
  : Unit = {
    // output: RGB RGB RGB | RGB RGB RGB
    // values: RGB RGB RGB | RGB RGB RGB
    input.foreachSample((off, length) => {
      ArrayEx.subtract(
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
      ArrayEx.subtract(
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
      ArrayEx.add(
        input.values, off, stride,
        -values(off),
        length
      )
    })
  }

  override protected def doPredictInvPerSample(input: RealArrayTensor)
  : Unit = {
    // output: RGB RGB RGB | RGB RGB RGB
    // values: R           | R
    input.foreachSamplePair((i, off, length) => {
      ArrayEx.add(
        input.values, off, 1,
        -values(i),
        length
      )
    })
  }

  override protected def doPredictInvPerBatch(input: RealArrayTensor)
  : Unit = {
    // output: RGB RGB RGB | RGB RGB RGB
    // values: R
    input += -values(0)
  }

}

object AddValues_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[AddValuesBuilder] {

  override def build(builder:        AddValuesBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : AddValues_JVM_Baseline = new AddValues_JVM_Baseline(
    builder, hints, seed, weightsBuilder
  )

}
