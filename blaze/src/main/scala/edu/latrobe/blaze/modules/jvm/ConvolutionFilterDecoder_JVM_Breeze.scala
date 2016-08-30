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

abstract class ConvolutionFilterDecoder_JVM_Breeze
  extends ConvolutionFilterDecoder_JVM {

  // ---------------------------------------------------------------------------
  //    Weights related
  // ---------------------------------------------------------------------------
  final override def refresh(): Unit = {}

  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  // We always do this in kernel mode. There is no point in forming a huge
  // matrix using the sparse code.
  final override protected def doDeriveFilterGradients(input:   RealArrayTensor,
                                                       context: PredictContext,
                                                       error:   RealArrayTensor,
                                                       sink:    RealArrayTensor)
  : Unit = {
    // Actual gradient
    val inp_t = input.valuesMatrix.t
    val err   = error.valuesMatrix
    val res   = sink.valuesMatrix

    kernel.foreachValidPair(outputSizeHint, noMaps, (i0, i1, offset0) => {
      //val n0   = i  * noMaps
      //val nEnd = n0 + noMaps
      val src_t = inp_t(::, i0 until i1)

      (j0, j1, offset0, offset1) => {
        /*
        val tmp = error(offset, ::)
        var w   = j
        var n   = n0
        while (n < nEnd) {
          // TODO: Why use update here? Could be faster (unsafeUpdate)!
          resultW(w) += tmp * in(n, ::).t
          /*
          if (gradientsW.stride == 1) {
            gradientsW.data(gradientsW.offset + w) += tmp * in(n, ::).t
          }
          else {
            gradientsW.unsafeUpdate(w, gradientsW.unsafeValueAt(w) + (tmp * in(n, ::).t))
            throw throw new InvalidOpenTypeException()
          }*/
          w          += kernel.noValues
          n          += 1
        }
        */
        val e   = err(offset0 until offset1, ::)
        val tmp = e * src_t
        val dst = res(j0 until j1, ::)
        dst += tmp
      }
    })
  }

}

final class ConvolutionFilterDecoder_JVM_Breeze_MM(override val builder:        ConvolutionFilterDecoderBuilder,
                                                   override val inputHints:     BuildHints,
                                                   override val seed:           InstanceSeed,
                                                   override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends ConvolutionFilterDecoder_JVM_Breeze {
  require(
    builder != null && inputHints != null && seed != null && weightBufferBuilder != null
  )


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input:  RealArrayTensor,
                                   output: RealArrayTensor)
  : PredictContext = {
    val inp = input.valuesMatrix
    val out = output.valuesMatrix

    // Normal weights
    kernel.foreachValidPair(outputSizeHint, noMaps, (i0, i1, offset0) => {
      /*val src = {
        val src0 = i * noMaps
        inp(src0 until src0 + noMaps, ::)
      }*/
      val src = inp(i0 until i1, ::)

      (j0, j1, offset0, offset1) => {
        //out(offset, ::) += wMat(j, ::) * src
        val fil = w(j0 until j1, ::)
        val dst = out(offset0 until offset1, ::)

        dst += fil * src
      }
    })

    EmptyContext
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(context:  PredictContext,
                                            oldError: RealArrayTensor,
                                            newError: RealArrayTensor)
  : Unit = {
    val oldErr = oldError.valuesMatrix
    val newErr = newError.valuesMatrix

    kernel.foreachValidPair(outputSizeHint, noMaps, (i0, i1, offset0) => {
      //val n0 = i * noMaps
      val dst = newErr(i0 until i1, ::)

      (j0, j1, offset0, offset1) => {
        /*
        val tmp = error(offset, ::).t
        var m   = 0
        while (m < noMaps) {
          // TODO: How about using unsafe method?
          //values(n0 + m, ::) += tmp * wMat.unsafeValueAt(j, m)
          axpy(wMat.unsafeValueAt(j, m), tmp, values(n0 + m, ::).t)
          m += 1
        }
        */
        val fil_t = w_t(::, j0 until j1)
        val err   = oldErr(offset0 until offset1, ::)
        dst += fil_t * err
      }
    })
  }

}

object ConvolutionFilterDecoder_JVM_Breeze_MM_Description
  extends ModuleVariant_JVM_Description[ConvolutionFilterDecoderBuilder] {

  override def build(builder:       ConvolutionFilterDecoderBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : ConvolutionFilterDecoder_JVM_Breeze_MM = new ConvolutionFilterDecoder_JVM_Breeze_MM(
    builder, hints, seed, weightsBuilder
  )

}

final class ConvolutionFilterDecoder_JVM_Breeze_SparseMM(override val builder:        ConvolutionFilterDecoderBuilder,
                                                         override val inputHints:     BuildHints,
                                                         override val seed:           InstanceSeed,
                                                         override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends ConvolutionFilterDecoder_JVM_Breeze {
  require(
    builder != null && inputHints != null && seed != null && weightBufferBuilder != null
  )


  // ---------------------------------------------------------------------------
  //    Weights related
  // ---------------------------------------------------------------------------
  /**
   * Expanded version of weights that makes computations faster most equations easier.
   */
  // It would be so nice, if we could do this without extra memory. However,
  // the current breeze implementation does not using array offsets.
  // (Hint: In this case, the iterators are the problem.. Fix if possible!)
  protected def generateWEx(inputSize: Size, outputSize: Size)
  : CSCMatrix[Real] = {
    val noInputs  = inputSize.noValues
    val noOutputs = outputSize.noValues
    val data = {
      val res = DenseMatrix.zeros[Real](
        filter.layout.noValues,
        noInputs / noMaps
      )
      res(::, *) := new DenseVector(filter.values)
      res.data
    }
    val rowIndices = {
      val res = DenseMatrix.zeros[Int](w.rows, noInputs)
      val tmp = DenseVector.zeros[Int](outputSize.noChannels)
      kernel.foreachValidPair(outputSize, noMaps, (i0, i1, offset0) => {
        val nRange = i0 until i1
        (j0, j1, offset0, offset1) => {
          cfor(offset0)(_ < offset1, _ + 1)(
            offset => tmp.data(offset - offset0) = offset0
          )
          val dst1 = res(j0 until j1, nRange)
          dst1(::, *) := tmp
        }
      })
      res.data
    }
    val colPointers = (0 to rowIndices.length by w.rows).toArray
    new CSCMatrix(data, noOutputs, noInputs, colPointers, rowIndices)
  }
  /*
  protected val wExData: DMat = DMat.zeros(kernel.noValues, noInputs)
  protected val wEx: SMat = {
    val colPointers = (0 to noInputs).map(i => /*wFlat.offset +*/ i * kernel.noValues).toArray
    val rowIndices = {
      val tmp = Array.ofDim[Int](wExData.size)
      kernel.foreachValidPair((i, offset) => {
        val w0 = i * noWeightsPerGroup

        (j, offset) => {
          var w = w0 + j
          var m = 0
          while (m < noMaps) {
            tmp(w) = offset
            w += kernel.noValues
            m += 1
          }
        }
      })
      tmp
    }
    new CSCMatrix(wExData.data, noOutputs, noInputs, colPointers, rowIndices)
  }
  update() // Do not remove!

  override def update(): Unit = {
    var m = 0
    while (m < wExData.cols) {
      val mNext = m + wMat.cols
      wExData(::, m until mNext) := wMat
      m = mNext
    }
  }
  */

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input:  RealArrayTensor,
                                   output: RealArrayTensor)
  : PredictContext ={
    val wEx = generateWEx(inputSize, outputSizeHint)
    val inp = input.valuesMatrix
    val out = output.valuesMatrix
    out := wEx * inp
    ConvolutionFilterDecoder_JVM_Breeze_SparseMM_Context(wEx)
  }

  /*
  override protected def doPredictInv(mode:    ComputeMode,
                                      output:  SampleTensor,
                                      context: Any)
  : SampleTensor = context match {
    case Convolution_Baseline_SparseContext(inpSize, wEx) =>
      var inp: DVec = output.values
      if (biasEnabled) {
        inp = inp - bias
      }
      inp = wEx.t * inp
      DenseSampleTensor(inp, inpSize)
  }*/


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  /*
  override def deriveInputError(mode:      ComputeMode,
                                input:     SampleTensor,
                                output:    SampleTensor,
                                context:   Any,
                                error:     SampleTensor,
                                reference: SampleTensor)
  : SampleTensor = context match {
    case Convolution_Baseline_SparseContext(inpSize, wEx) =>
      DenseSampleTensor(wEx.t * error.values, input.size)
    case _ =>
      val wEx = generateMatrix(input.size, output.size)
      DenseSampleTensor(wEx.t * error.values, input.size)
  }*/

  override protected def doDeriveInputError(context:  PredictContext,
                                            oldError: RealArrayTensor,
                                            newError: RealArrayTensor)
  : Unit = context match {
    case ConvolutionFilterDecoder_JVM_Breeze_SparseMM_Context(wEx) =>
      val oldErr = oldError.valuesMatrix
      val newErr = newError.valuesMatrix
      newErr := wEx.t * oldErr
    case _ =>
      throw new MatchError(context)
  }

}

final case class ConvolutionFilterDecoder_JVM_Breeze_SparseMM_Context(wEx: CSCMatrix[Real])
  extends PredictContext {
}

object ConvolutionFilterDecoder_JVM_Breeze_SparseMM_Description
  extends ModuleVariant_JVM_Description[ConvolutionFilterDecoderBuilder] {

  override def build(builder:        ConvolutionFilterDecoderBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : ConvolutionFilterDecoder_JVM_Breeze_SparseMM = new ConvolutionFilterDecoder_JVM_Breeze_SparseMM(
    builder, hints, seed, weightsBuilder
  )

}
