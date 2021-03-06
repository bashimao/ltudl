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

final class LinearFilterDecoder_JVM_Breeze(override val builder:        LinearFilterDecoderBuilder,
                                           override val inputHints:     BuildHints,
                                           override val seed:           InstanceSeed,
                                           override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends LinearFilterDecoder_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input: RealArrayTensor)
  : RealArrayTensor = {
    val inp = input.valuesMatrix
    val out = w * inp
    RealArrayTensor.derive(outputSize, out)
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveFilterGradients(input: RealArrayTensor,
                                                 error: RealArrayTensor,
                                                 sink:  RealArrayTensor)
  : Unit = {
    val inp = input.valuesMatrix
    val err = error.valuesMatrix
    val dst = sink.valuesMatrix
    val tmp = err * inp.t
    dst += tmp
  }

  override protected def doDeriveInputError(error: RealArrayTensor)
  : RealArrayTensor = {
    val oldErr = error.valuesMatrix
    val newErr = w_t * oldErr
    RealArrayTensor.derive(inputSizeHint, newErr)
  }

}

object LinearFilterDecoder_JVM_Breeze_Description
  extends ModuleVariant_JVM_Description[LinearFilterDecoderBuilder] {

  override def build(builder:        LinearFilterDecoderBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : LinearFilterDecoder_JVM_Breeze = new LinearFilterDecoder_JVM_Breeze(
    builder, hints, seed, weightsBuilder
  )

}