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

abstract class ConvolutionFilter_JVM_BLAS
  extends ConvolutionFilter_JVM {

  // ---------------------------------------------------------------------------
  //    Weights related
  // ---------------------------------------------------------------------------
  final override def refresh(): Unit = {}

}

// TODO: Further reduce overhead!
final class ConvolutionFilter_JVM_BLAS_MM(override val builder:        ConvolutionFilterBuilder,
                                          override val inputHints:     BuildHints,
                                          override val seed:           InstanceSeed,
                                          override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends ConvolutionFilter_JVM_BLAS {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input:  RealArrayTensor,
                                   output: RealArrayTensor)
  : PredictContext = {
    val inpSize = input.layout.size
    val inp     = input.valuesMatrix
    val out     = output.valuesMatrix

    // Filter
    kernel.foreachValidPair(inpSize, noMaps, (i0, i1, offset0) => {
      val dst = out(i0 until i1, ::)

      (j0, j1, offset0, offset1) => {
        val fil_t = w_t(::, j0 until j1)
        val src   = inp(offset0 until offset1, ::)

        _BLAS.gemm(
          Real.one,
          fil_t,
          src,
          Real.one,
          dst
        )
      }
    })

    EmptyContext
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveFilterGradients(input:   RealArrayTensor,
                                                 context: PredictContext,
                                                 error:   RealArrayTensor,
                                                 sink:    RealArrayTensor)
  : Unit = {
    val inpSize = input.layout.size
    val inp     = input.valuesMatrix
    val err_t   = error.valuesMatrix.t
    val res     = sink.valuesMatrix

    kernel.foreachValidPair(inpSize, noMaps, (i0, i1, offset0) => {
      val e_t = err_t(::, i0 until i1)

      (j0, j1, offset0, offset1) => {
        val src = inp(offset0 until offset1, ::)
        val dst = res(j0 until j1, ::)
        _BLAS.gemm(
          Real.one,
          src,
          e_t,
          Real.one,
          dst
        )
      }
    })
  }

  override protected def doDeriveInputError(context:  PredictContext,
                                            oldError: RealArrayTensor,
                                            newError: RealArrayTensor)
  : Unit = {
    val oldErr  = oldError.valuesMatrix
    val newErr  = newError.valuesMatrix
    val inpSize = newError.layout.size

    kernel.foreachValidPair(inpSize, noMaps, (i0, i1, offset0) => {
      val err = oldErr(i0 until i1, ::)

      (j0, j1, offset0, offset1) => {
        val fil = w(j0 until j1, ::)
        val dst = newErr(offset0 until offset1, ::)

        _BLAS.gemm(
          Real.one,
          fil,
          err,
          Real.one,
          dst
        )
      }
    })
  }

}

object ConvolutionFilter_JVM_BLAS_MM_Description
  extends ModuleVariant_JVM_Description[ConvolutionFilterBuilder] {

  override def build(builder:        ConvolutionFilterBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : ConvolutionFilter_JVM_BLAS_MM = new ConvolutionFilter_JVM_BLAS_MM(
    builder, hints, seed, weightsBuilder
  )

}

// TODO: Further reduce overhead!
final class ConvolutionFilter_JVM_BLAS_ImplicitMM(override val builder:        ConvolutionFilterBuilder,
                                                  override val inputHints:     BuildHints,
                                                  override val seed:           InstanceSeed,
                                                  override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends ConvolutionFilter_JVM_BLAS {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input: RealArrayTensor,
                                   output: RealArrayTensor)
  : PredictContext = {
    val inpSize = input.layout.size
    val inp     = input.valuesMatrix
    val out     = output.valuesMatrix

    kernel.foreachValidPair(inpSize, noMaps, (i0, i1, offset0) => {
      val m       = i1 - i0
      val oOffset = out.linearIndex(i0, 0)

      (j0, j1, offset0, offset1) => {
        val n       = j1 - j0
        val wOffset = w_t.linearIndex(0, j0)
        val iOffset = inp.linearIndex(offset0, 0)

        _BLAS.gemm(
          Real.one,
          w_t, wOffset, w_t.rows, n,
          inp, iOffset, n,        inp.cols,
          Real.one,
          out, oOffset, m,        out.cols
        )
      }
    })

    EmptyContext
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveFilterGradients(input:   RealArrayTensor,
                                                 context: PredictContext,
                                                 error:   RealArrayTensor,
                                                 sink:    RealArrayTensor)
  : Unit = {
    val inpSize = input.layout.size
    val inp     = input.valuesMatrix
    val e_t     = error.valuesMatrix.t
    val res     = sink.valuesMatrix

    kernel.foreachValidPair(inpSize, noMaps, (i0, i1, offset0) => {
      val m       = i1 - i0
      val eOffset = e_t.linearIndex(0, i0)

      (j0, j1, offset0, offset1) => {
        val n       = j1 - j0
        val iOffset = inp.linearIndex(offset0, 0)
        val rOffset = res.linearIndex(j0, 0)

        _BLAS.gemm(
          Real.one,
          inp, iOffset, n,        inp.cols,
          e_t, eOffset, e_t.rows, m,
          Real.one,
          res, rOffset, n,        res.cols
        )
      }
    })
  }

  override protected def doDeriveInputError(context:  PredictContext,
                                            oldError: RealArrayTensor,
                                            newError: RealArrayTensor)
  : Unit = {
    val oldErr  = oldError.valuesMatrix
    val newErr  = newError.valuesMatrix
    val inpSize = newError.layout.size

    kernel.foreachValidPair(inpSize, noMaps, (i0, i1, offset0) => {
      val m       = i1 - i0
      val oOffset = oldErr.linearIndex(i0, 0)

      (j0, j1, offset0, offset1) => {
        val n       = j1 - j0
        val wOffset = w.linearIndex(j0, 0)
        val nOffset = newErr.linearIndex(offset0, 0)

        _BLAS.gemm(
          Real.one,
          w,      wOffset, n, w.cols,
          oldErr, oOffset, m, oldErr.cols,
          Real.one,
          newErr, nOffset, n, newErr.cols
        )
      }
    })
  }

}

object ConvolutionFilter_JVM_BLAS_ImplicitMM_Description
  extends ModuleVariant_JVM_Description[ConvolutionFilterBuilder] {

  override def build(builder:        ConvolutionFilterBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : ConvolutionFilter_JVM_BLAS_ImplicitMM = new ConvolutionFilter_JVM_BLAS_ImplicitMM(
    builder, hints, seed, weightsBuilder
  )

}