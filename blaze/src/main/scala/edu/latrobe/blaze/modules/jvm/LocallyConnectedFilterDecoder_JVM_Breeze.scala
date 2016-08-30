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

import breeze.linalg.{*, CSCMatrix, DenseMatrix, DenseVector}
import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import spire.implicits._

abstract class LocallyConnectedFilterDecoder_JVM_Breeze
  extends LocallyConnectedFilterDecoder_JVM {

  // We always do this in kernel mode. There is no point in forming a huge
  // matrix using the sparse code.
  final override protected def doDeriveFilterGradients(input: RealArrayTensor,
                                                       error: RealArrayTensor,
                                                       sink:  RealArrayTensor)
  : Unit = {
    //error.valuesMatrix.addCols(sink)

    //val offset2    = offset1 + wFlat.length
    //val gradientsW = result(offset1 until offset2)

    // This avoid creating a huge matrix of unmanageable size which we would
    // anyway not need.
    val inp_t = input.valuesMatrix.t
    val err   = error.valuesMatrix
    val dst   = sink.valuesMatrix

    kernel.foreachValidPair(outputSize, noMaps, (i0, i1, offset0) => {
      val iRange = i0 until i1
      /*val n0   = i  * noMaps
      val nEnd = n0 + noMaps
      */
      // TODO: Why not use w vectorized?
      //val w0   = n0 * kernel.noValues
      val r     = dst(::, iRange)
      val src_t = inp_t(::, iRange)

      (j0, j1, offset0, offset1) => {
        /*
        val tmp = error(offset, ::)
        var w   = w0 + j
        var n   = n0
        while (n < nEnd) {
          // TODO: Why use update?
          // TODO: Use other dot function.
          //test.data(w0 + m) = a dot rawError(n0 + m, ::)
          resultW.unsafeUpdate(w, tmp * in(n, ::).t)
          w += kernel.noValues
          n += 1
        }
        */
        val e   = err(offset0 until offset1, ::)
        val tmp = e * src_t
        val dst = r(j0 until j1, ::)
        dst += tmp
      }
    })
    /*
    // TODO: Have to find a way to do this better without wasting too much memory.
    // TODO: Just to avoid as single allocation? Isn't that a little bit much effort?
    if (lambda._1.isNaN && lambda._2.isNaN) {
      kernel.foreachPair((i, offset) => {
        val n0   = i  * noMaps
        val nEnd = n0 + noMaps
        // TODO: Why not use w vectorized?
        val w0   = n0 * kernel.size

        (j, offset) => {
          val tmp = rawError(offset, ::)
          var w   = w0 + j
          var n   = n0
          while (n < nEnd) {
            // TODO: Why use update?
            // TODO: Use other dot function.
            //test.data(w0 + m) = a dot rawError(n0 + m, ::)
            gradientsW.update(w, tmp * in(n, ::).t)
            w += kernel.size
            n += 1
          }
        }
      })
    }
    else if (lambda._1.isNaN) {
      gradientsW := wFlat
      gradientsW *= lambda._2 * rawError.cols
      kernel.foreachPair((i, offset) => {
        val n0   = i  * noMaps
        val nEnd = n0 + noMaps
        // TODO: Why not use w vectorized?
        val w0   = n0 * kernel.size

        (j, offset) => {
          val tmp = rawError(offset, ::)
          var w   = w0 + j
          var n   = n0
          while (n < nEnd) {
            gradientsW(w) += tmp * in(n, ::).t
            w             += kernel.size
            n             += 1
          }
        }
      })
    }
    else if (lambda._2.isNaN) {
      gradientsW := wFlat
      gradientsW *= lambda._1 * rawError.cols
      val tmp2 = wFlat :* wFlat
      tmp2 += epsilon
      sqrt.inPlace(tmp2)
      gradientsW :/= tmp2
      kernel.foreachPair((i, offset) => {
        val n0   = i  * noMaps
        val nEnd = n0 + noMaps
        // TODO: Why not use w vectorized?
        val w0   = n0 * kernel.size

        (j, offset) => {
          val tmp = rawError(offset, ::)
          var w   = w0 + j
          var n   = n0
          while (n < nEnd) {
            gradientsW(w) += tmp * in(n, ::).t
            w             += kernel.size
            n             += 1
          }
        }
      })
    }
    else {
      gradientsW := wFlat
      gradientsW *= lambda._1 * rawError.cols
      val tmp2 = wFlat :* wFlat
      tmp2 += epsilon
      sqrt.inPlace(tmp2)
      gradientsW :/= tmp2
      gradientsW +=  wFlat * (lambda._2 * rawError.cols)
      kernel.foreachPair((i, offset) => {
        val n0   = i  * noMaps
        val nEnd = n0 + noMaps
        // TODO: Why not use w vectorized?
        val w0   = n0 * kernel.size

        (j, offset) => {
          val tmp = rawError(offset, ::)
          var w   = w0 + j
          var n   = n0
          while (n < nEnd) {
            gradientsW(w) += tmp * in(n, ::).t
            w += kernel.size
            n += 1
          }
        }
      })
    }*/
  }

}

final class LocallyConnectedFilterDecoder_JVM_Breeze_MM(override val builder:        LocallyConnectedFilterDecoderBuilder,
                                                        override val inputHints:     BuildHints,
                                                        override val seed:           InstanceSeed,
                                                        override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends LocallyConnectedFilterDecoder_JVM_Breeze {
  require(
    builder != null && inputHints != null && seed != null && weightBufferBuilder != null
  )


  // ---------------------------------------------------------------------------
  //    Weights related
  // ---------------------------------------------------------------------------
  override def refresh()
  : Unit = {}


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input:  RealArrayTensor,
                                   output: RealArrayTensor)
  : Unit = {
    val inp = input.valuesMatrix
    val out = output.valuesMatrix

    // Filter
    kernel.foreachValidPair(outputSize, noMaps, (i0, i1, offset0) => {
      //val n0  = i * noMaps
      //val rng = n0 until (n0 + noMaps)
      val iRange = i0 until i1
      val src    = inp(iRange, ::)
      val wSlice = w(::, iRange)

      (j0, j1, offset0, offset1) => {
        // TODO: Think about a way that avoids this allocation and does not break optimizations.
        //out(offset, ::) += wMat(j, rng) * src
        val w   = wSlice(j0 until j1, ::)
        val tmp = w * src
        val dst = out(offset0 until offset1, ::)
        dst += tmp
      }
    })
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(oldError: RealArrayTensor,
                                            newError: RealArrayTensor)
  : Unit = {
    val oldErr = newError.valuesMatrix
    val newErr = newError.valuesMatrix

    kernel.foreachValidPair(outputSize, noMaps, (i0, i1, offset0) => {
      //val n0  = i * noMaps
      //val rng = n0 until (n0 + noMaps)
      val iRange   = i0 until i1
      val dst      = newErr(iRange, ::)
      val wSlice_t = w_t(iRange, ::)

      (j0, j1, offset0, offset1) => {
        // TODO: Think about a way that avoids this allocation and does not break optimizations.
        // TODO: Use axpy!
        //dst += wMat(j, nRange).t * error(offset, ::)
        val w_t = wSlice_t(::, j0 until j1)
        val err = oldErr(offset0 until offset1, ::)
        dst += w_t * err
      }
    })
  }

}

object LocallyConnectedFilterDecoder_JVM_Breeze_MM_Description
  extends ModuleVariant_JVM_Description[LocallyConnectedFilterDecoderBuilder] {

  override def build(builder:       LocallyConnectedFilterDecoderBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : LocallyConnectedFilterDecoder_JVM_Breeze_MM = {
    new LocallyConnectedFilterDecoder_JVM_Breeze_MM(
      builder, hints, seed, weightsBuilder
    )
  }

}

final class LocallyConnectedFilterDecoder_JVM_Breeze_SparseMM(override val builder:        LocallyConnectedFilterDecoderBuilder,
                                                              override val inputHints:     BuildHints,
                                                              override val seed:           InstanceSeed,
                                                              override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends LocallyConnectedFilterDecoder_JVM_Breeze {
  require(
    builder != null && inputHints != null && seed != null && weightBufferBuilder != null
  )

  // ---------------------------------------------------------------------------
  //    Weights related.
  // ---------------------------------------------------------------------------
  assume(w.offset == 0)

  private val wEx: CSCMatrix[Real] = {
    val rowIndices = {
      val res = DenseMatrix.zeros[Int](w.rows, w.cols)
      val tmp = DenseVector.zeros[Int](outputSize.noChannels)
      kernel.foreachValidPair(outputSize, noMaps, (i0, i1, offset0) => {
        //val n0 = i * noWeightsPerGroup
        val nRange = i0 until i1
        (j0, j1, offset0, offset1) => {
          cfor(offset0)(_ < offset1, _ + 1)(
            offset => tmp.data(offset - offset0) = offset0
          )
          val dst = res(j0 until j1, nRange)
          dst(::, *) := tmp
        }
      })
      res.data
    }
    val colPointers = (0 to rowIndices.length by w.rows).toArray
    new CSCMatrix(
      w.data,
      filter.layout.noValues,
      w.cols,
      colPointers,
      rowIndices
    )
  }
  /*
  protected val wExData: DVec = DVec.zeros(w.length)
  protected val wEx: SMat = {
    //val colPointers = (0 to noInputs).map(i => /*wFlat.offset +*/ i * kernel.noValues).toArray
    val colPointers = (0 to w.length by wMat.rows).toArray
    val rowIndices = {
      val res = DenseMatrix.zeros[Int](wMat.rows, wMat.cols)
      val tmp = DenseVector.zeros[Int](outputSize.noChannels)
      kernel.foreachValidPair(outputSize, (i0, i1, offset) => {
        //val w0 = i * noWeightsPerGroup
        val nRange = i0 until i1
        (j0, j1, offset0, offset1) => {
          /*
            var w = w0 + j
            var m = 0
            while (m < noMaps) {
              tmp(w) = offset
              w += kernel.noValues
              m += 1
            }
          }
          */
          cfor(offset0)(_ < offset1, _ + 1)(
            offset => tmp.data(offset - offset0) = offset0
          )
          val dst = res(j0 until j1, nRange)
          dst(::, *) := tmp
        }
      })
      res.data
    }
    new CSCMatrix(wExData.data, noOutputs, noInputs, colPointers, rowIndices)
  }
  update() // Do not remove!


  override def update(): Unit = wExData := w
  */

  private var wEx_t
  : CSCMatrix[Real] = _

  override def refresh()
  : Unit = {
    wEx_t = wEx.t
  }


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input:  RealArrayTensor,
                                   output: RealArrayTensor)
  : Unit = {
    val inp = input.valuesMatrix
    val out = output.valuesMatrix
    out := wEx * inp
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(oldError: RealArrayTensor,
                                            newError: RealArrayTensor)
  : Unit = {
    val oldErr = oldError.valuesMatrix
    val newErr = newError.valuesMatrix
    newErr := wEx_t * oldErr
  }

}

final case class LocallyConnectedFilterDecoder_JVM_Breeze_SparseMM_Context(inputSize:      Size,
                                                                           filterMatrixEx: CSCMatrix[Real])
  extends PredictContext {
}

object LocallyConnectedFilterDecoder_JVM_Breeze_SparseMM_Description
  extends ModuleVariant_JVM_Description[LocallyConnectedFilterDecoderBuilder] {

  override def build(builder:        LocallyConnectedFilterDecoderBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : LocallyConnectedFilterDecoder_JVM_Breeze_SparseMM = {
    new LocallyConnectedFilterDecoder_JVM_Breeze_SparseMM(
      builder, hints, seed, weightsBuilder
    )
  }

}
