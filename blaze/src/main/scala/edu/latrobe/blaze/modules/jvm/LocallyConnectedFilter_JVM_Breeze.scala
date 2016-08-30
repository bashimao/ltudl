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

abstract class LocallyConnectedFilter_JVM_Breeze
  extends LocallyConnectedFilter_JVM {

  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  // We always do this in kernel mode. There is no point in forming a huge
  // matrix using the sparse code.
  final override protected def doDeriveFilterGradients(input: RealArrayTensor,
                                                       error: RealArrayTensor,
                                                       sink:  RealArrayTensor)
  : Unit = {
    // Compute gradients and regularize all but the bias weights.
    // Compute and regularize remaining weights.
    //val offset2 = offset1 + wFlat.length
    //val gradientsW = result(offset1 until offset2)

    // Nice for error checking: Should contain similar values.
    val inpSize = input.layout.size
    val inp     = input.valuesMatrix
    val err_t   = error.valuesMatrix.t
    val dst     = sink.valuesMatrix

    kernel.foreachValidPair(inpSize, noMaps, (i0, i1, offset0) => {
      val iRange = i0 until i1
      /*val w0 = i * noWeightsPerGroup
    resultW(w0 until w0 + noWeightsPerGroup).asMatrix(
      kernel.noValues, noMaps
    )*/
      /*val err = {
      val n0 = i * noMaps
      error(n0 until n0 + noMaps, ::).t
    }*/
      val r   = dst(::, iRange)
      val e_t = err_t(::, iRange)

      (j0, j1, offset0, offset1) => {
        //val src = inp(offset, ::)
        // TODO: Lots of small allocations!
        //dst(j, ::) := src * err
        val src = inp(offset0 until offset1, ::)
        val tmp = src * e_t
        val dst = r(j0 until j1, ::)
        dst += tmp
      }
      /*
    val n0   = i  * noMaps
    val nEnd = n0 + noMaps
    val w0   = n0 * kernel.size

    (j, offset) => {
      val a = in(offset, ::)
      var w = w0 + j
      var n = n0
      while (n < nEnd) {
        gradientsW.unsafeUpdate(w, a * rawError(n, ::).t)
        w += kernel.size
        n += 1
      }
    }*/
    })
    /*
  // TODO: Have to find a way to do this better without wasting too much memory.
  // TODO: Just to avoid as single allocation? Isn't that a little bit much effort?
  if (lambda._1.isNaN && lambda._2.isNaN) {
    kernel.foreachPair((i, offset) => {
      val n0   = i  * noMaps
      val nEnd = n0 + noMaps
      val w0   = n0 * kernel.size

      (j, offset) => {
        val a = in(offset, ::)
        var w = w0 + j
        var n = n0
        while (n < nEnd) {
          gradientsW.update(w, a * rawError(n, ::).t)
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
      val w0   = n0 * kernel.size

      (j, offset) => {
        val a = in(offset, ::)
        var w = w0 + j
        var n = n0
        while (n < nEnd) {
          gradientsW(w) += a * rawError(n, ::).t
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
      val w0   = n0 * kernel.size

      (j, offset) => {
        val a = in(offset, ::)
        var w = w0 + j
        var n = n0
        while (n < nEnd) {
          gradientsW(w) += a * rawError(n, ::).t
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
      val w0   = n0 * kernel.size

      (j, offset) => {
        val a = in(offset, ::)
        var w = w0 + j
        var n = n0
        while (n < nEnd) {
          gradientsW(w) += a * rawError(n, ::).t
          w             += kernel.size
          n             += 1
        }
      }
    })
  }
  */
  }

}

final class LocallyConnectedFilter_JVM_Breeze_MM(override val builder:        LocallyConnectedFilterBuilder,
                                                 override val inputHints:     BuildHints,
                                                 override val seed:           InstanceSeed,
                                                 override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends LocallyConnectedFilter_JVM_Breeze {

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
    val inpSize = input.layout.size
    val inp     = input.valuesMatrix
    val out     = output.valuesMatrix

    // Filter
    kernel.foreachValidPair(inpSize, noMaps, (i0, i1, offset0) => {
      // TODO: Could precompute these ranges.
      /*
      val nRange = {
        val n0 = i * noMaps
        n0 until n0 + noMaps
      }
      */
      //val n0 = i * noMaps
      //val n1 = n0 + noMaps
      val iRange   = i0 until i1
      val dst      = out(iRange, ::)
      val wSlice_t = w_t(iRange, ::)

      (j0, j1, offset0, offset1) => {
        /*
        // TODO: Replace this load with vectorization?!
        cfor(offset0)(_ < offset1, _ + 1)(offset => {
          val src = inp(offset, ::).t
          // TODO: Think about a way that avoids this allocation and does not break optimizations.
          //dst += wMat(j, nRange).t * inp(offset, ::)
          // TODO: Still some optimization potential left.
          cfor(i0)(_ < i1, _ + 1)(
            n => axpy(wMat.unsafeValueAt(j, n), src, out(n, ::).t)
          )
        })*/
        val w_t = wSlice_t(::, j0 until j1)
        val src = inp(offset0 until offset1, ::)
        dst += w_t * src
      }
    })
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(oldError: RealArrayTensor,
                                            newError: RealArrayTensor)
  : Unit = {
    val oldErr = oldError.valuesMatrix
    val newErr = newError.valuesMatrix

    kernel.foreachValidPair(inputSizeHint, noMaps, (i0, i1, offset0) => {
      //val n0   = i  * noMaps
      //val nEnd = n0 + noMaps
      val iRange = i0 until i1
      val err    = oldErr(iRange, ::)
      val wSlice = w(::, iRange)

      (j0, j1, offset0, offset1) => {
        /*
        // TODO: Replace this load with vectorization?!
        cfor(offset0)(_ < offset1, _ + 1)(offset => {
          val tmp = values(offset, ::).t
          cfor(i0)(_ < i1, _ + 1)(n =>
            // TODO: How about using unsafe method?
            // TODO: Performance; Why don't we do that in one step?
            // TODO: There is much performance gained with new iterator method.
            axpy(wMat.unsafeValueAt(j, n), error(n, ::).t, tmp)
          )
        })
        */
        val w   = wSlice(j0 until j1, ::)
        val tmp = w * err
        val dst = newErr(offset0 until offset1, ::)
        dst += tmp
      }
    })

    RealArrayTensor.derive(inputSizeHint, newErr)
  }

}

object LocallyConnectedFilter_JVM_Breeze_MM_Description
  extends ModuleVariant_JVM_Description[LocallyConnectedFilterBuilder] {

  override def build(builder:        LocallyConnectedFilterBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : LocallyConnectedFilter_JVM_Breeze_MM = {
    new LocallyConnectedFilter_JVM_Breeze_MM(
      builder, hints, seed, weightsBuilder
    )
  }

}

final class LocallyConnectedFilter_JVM_Breeze_SparseMM(override val builder:        LocallyConnectedFilterBuilder,
                                                       override val inputHints:     BuildHints,
                                                       override val seed:           InstanceSeed,
                                                       override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends LocallyConnectedFilter_JVM_Breeze {

  // ---------------------------------------------------------------------------
  //    Weights related
  // ---------------------------------------------------------------------------
  assume(w.offset == 0)

  private val wEx: CSCMatrix[Real] = {
    val rowIndices = {
      val res = DenseMatrix.zeros[Int](w.rows, w.cols)
      val tmp = DenseVector.zeros[Int](inputSizeHint.noChannels)
      kernel.foreachValidPair(inputSizeHint, noMaps, (i0, i1, offset0) => {
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

  private var wEx_t: CSCMatrix[Real] = null

  override def refresh(): Unit = {
    wEx_t = wEx.t
  }

  /*
  protected val wExData: DVec = DVec.zeros(w.length)
  protected val wEx: SMat = {
    //val colPointers = (0 to noOutputs).map(i => /*wFlat.offset +*/ i * kernel.size).toArray
    val colPointers = (0 to w.length by wMat.rows).toArray
    val rowIndices = {
      val res = DenseMatrix.zeros[Int](wMat.rows, wMat.cols)
      val tmp = DenseVector.zeros[Int](inputSize.noChannels)
      kernel.foreachValidPair(inputSize, (i0, i1, offset0) => {
        //val n0 = i * noWeightsPerGroup
        val nRange = i0 until i1
        (j0, j1, offset0, offset1) => {
          /*
          var n = n0 + j
          var m = 0
          while (m < noMaps) {
            rowIndices(n) = offset
            n += kernel.noValues
            m += 1
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
    new CSCMatrix(wExData.data, noInputs, noOutputs, colPointers, rowIndices)
  }
  update() // Do not remove!

  override def update(): Unit = wExData := w
  */


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input:  RealArrayTensor,
                                   output: RealArrayTensor)
  : Unit = {
    val inp = input.valuesMatrix
    val out = output.valuesMatrix
    out := wEx_t * inp
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(oldError: RealArrayTensor,
                                            newError: RealArrayTensor)
  : Unit = {
    val oldErr = oldError.valuesMatrix
    val newErr = newError.valuesMatrix
    newErr := wEx * oldErr
  }

}

object LocallyConnectedFilter_JVM_Breeze_SparseMM_Description
  extends ModuleVariant_JVM_Description[LocallyConnectedFilterBuilder] {

  override def build(builder:        LocallyConnectedFilterBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : LocallyConnectedFilter_JVM_Breeze_SparseMM = {
    new LocallyConnectedFilter_JVM_Breeze_SparseMM(
      builder, hints, seed, weightsBuilder
    )
  }

}
