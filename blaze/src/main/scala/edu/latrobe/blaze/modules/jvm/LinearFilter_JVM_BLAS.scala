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
import edu.latrobe.blaze.modules.LinearFilterBuilder

/**
 * @param builder The description object that defines the properties of the layer.
 * @param weightBufferBuilder Context that keeps track of parameter bindings.
 */
final class LinearFilter_JVM_BLAS(override val builder:        LinearFilterBuilder,
                                  override val inputHints:     BuildHints,
                                  override val seed:           InstanceSeed,
                                  override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends LinearFilter_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input:  RealArrayTensor,
                                   output: RealArrayTensor)
  : Unit = {
    val inp = input.valuesMatrix
    val out = output.valuesMatrix
    _BLAS.gemm(
      Real.one,
      w_t,
      inp,
      Real.zero,
      out
    )
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
    _BLAS.gemm(
      Real.one,
      inp,
      err.t,
      Real.one,
      dst
    )
  }

  override protected def doDeriveInputError(oldError: RealArrayTensor,
                                            newError: RealArrayTensor)
  : Unit = {
    val oldErr = oldError.valuesMatrix
    val newErr = newError.valuesMatrix
    _BLAS.gemm(
      Real.one,
      w,
      oldErr,
      Real.zero,
      newErr
    )
  }

}

object LinearFilter_JVM_BLAS_Description
  extends ModuleVariant_JVM_Description[LinearFilterBuilder] {

  override def build(builder:        LinearFilterBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : LinearFilter_JVM_BLAS = new LinearFilter_JVM_BLAS(
    builder, hints, seed, weightsBuilder
  )

}