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

import edu.latrobe.blaze._
import edu.latrobe.blaze.modules.MeanBuilder
import edu.latrobe.{RealArrayTensor, _}

final class Mean_JVM_Baseline(override val builder:        MeanBuilder,
                              override val inputHints:     BuildHints,
                              override val seed:           InstanceSeed,
                              override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Mean_JVM  {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredictPerUnit(input:  RealArrayTensor,
                                          output: RealArrayTensor)
  : Unit = {
    input.foreachUnit((off, stride, length) => {
      val tmp = ArrayEx.mean(
        input.values, off, stride,
        length
      )
      output.values(off) = tmp
    })
  }

  override protected def doPredictPerChannel(input:  RealArrayTensor,
                                             output: RealArrayTensor)
  : Unit = {
    input.foreachChannel((off, stride, length) => {
      val tmp = ArrayEx.mean(
        input.values, off, stride,
        length
      )
      output.values(off) = tmp
    })
  }

  override protected def doPredictPerSample(input:  RealArrayTensor,
                                            output: RealArrayTensor)
  : Unit = {
    input.foreachSamplePair((i, off, length) => {
      val tmp = ArrayEx.mean(
        input.values, off, 1,
        length
      )
      output.values(i) = tmp
    })
  }

  override protected def doPredictPerBatch(input:  RealArrayTensor,
                                           output: RealArrayTensor)
  : Unit = {
    output.values(0) = input.mean
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------

  override protected def doDeriveInputErrorPerUnit(oldError: RealArrayTensor,
                                                   newError: RealArrayTensor)
  : Unit = {
    newError.foreachUnit((off, stride, length) => {
      ArrayEx.fill(
        newError.values, off, stride,
        oldError.values(off) / length,
        length
      )
    })
    /*
    // Scale down.
    oldError *= Real.one / newError.layout.noSamples

    // Distribute error.
    val oldErr = oldError.values
    val newErr = newError.values
    newError.foreachSample(off0 => {
      Array.copy(
        oldErr, 0,
        newErr, off0,
        oldErr.length
      )
    })
    */
  }

  override protected def ddoDeriveInputErrorPerChannel(oldError: RealArrayTensor,
                                                       newError: RealArrayTensor)
  : Unit = {
    newError.foreachChannel((off, stride, length) => {
      ArrayEx.fill(
        newError.values, off, stride,
        oldError.values(off) / length,
        length
      )
    })
  }

  override protected def doDeriveInputErrorPerSample(oldError: RealArrayTensor,
                                                     newError: RealArrayTensor)
  : Unit = {
    newError.foreachSamplePair((i, off, length) => {
      ArrayEx.fill(
        newError.values, off, 1,
        oldError.values(i) / length,
        length
      )
    })
  }

  override protected def doDeriveInputErrorPerBatch(oldError: RealArrayTensor,
                                                    newError: RealArrayTensor)
  : Unit = {
    newError := oldError.values(0) / newError.layout.noValues
  }

}

object Mean_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[MeanBuilder] {

  override def build(builder:        MeanBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Mean_JVM_Baseline = new Mean_JVM_Baseline(
    builder, hints, seed, weightsBuilder
  )

}