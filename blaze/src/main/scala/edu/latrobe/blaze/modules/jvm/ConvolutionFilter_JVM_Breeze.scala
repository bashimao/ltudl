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

abstract class ConvolutionFilter_JVM_Breeze
  extends ConvolutionFilter_JVM {

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
    val inpSize = input.layout.size
    val inp     = input.valuesMatrix
    val err     = error.valuesMatrix
    val err_t   = err.t
    val res     = sink.valuesMatrix

    kernel.foreachValidPair(inpSize, noMaps, (i0, i1, offset0) => {
      val e_t = err_t(::, i0 until i1)

      (j0, j1, offset0, offset1) => {
        val src = inp(offset0 until offset1, ::)
        val tmp = src * e_t
        val dst = res(j0 until j1, ::)
        dst += tmp
      }
    })
  }

}

final class ConvolutionFilter_JVM_Breeze_MM(override val builder:        ConvolutionFilterBuilder,
                                            override val inputHints:     BuildHints,
                                            override val seed:           InstanceSeed,
                                            override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends ConvolutionFilter_JVM_Breeze {
  require(builder        != null)
  require(inputHints     != null)
  require(seed           != null)
  require(weightBufferBuilder != null)

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input:  RealArrayTensor,
                                   output: RealArrayTensor)
  : PredictContext = {
    val inpSize = input.layout.size
    val inp     = input.valuesMatrix
    val out     = output.valuesMatrix
    val fil_t   = w_t

    // Filter
    kernel.foreachValidPair(inpSize, noMaps, (i0, i1, offset0) => {
      val dst = out(i0 until i1, ::)

      (j0, j1, offset0, offset1) => {
        val w_t = fil_t(::, j0 until j1)
        val src = inp(offset0 until offset1, ::)
        val tmp = w_t * src
        dst += tmp
      }
    })

    EmptyContext
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(context: PredictContext,
                                            oldError: RealArrayTensor,
                                            newError: RealArrayTensor)
  : Unit = {
    val inpSize = newError.layout.size
    val oldErr  = oldError.valuesMatrix
    val newErr  = newError.valuesMatrix
    val fil     = w

    kernel.foreachValidPair(inpSize, noMaps, (i0, i1, offset0) => {
      val err = oldErr(i0 until i1, ::)

      (j0, j1, offset0, offset1) => {
        val w   = fil(j0 until j1, ::)
        val dst = newErr(offset0 until offset1, ::)

        dst += w * err
      }
    })
  }

}

object ConvolutionFilter_JVM_Breeze_MM_Description
  extends ModuleVariant_JVM_Description[ConvolutionFilterBuilder] {

  override def build(builder:        ConvolutionFilterBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : ConvolutionFilter_JVM_Breeze_MM = new ConvolutionFilter_JVM_Breeze_MM(
    builder, hints, seed, weightsBuilder
  )

}

/**
 * Each map encodes a feature. The maximum size of the feature is limited
 * by the kernel size.
 *
 * Each map is a kernel that is "mapDims"-times pulled over the input. The
 * layer output is contains the weighted inputs in an interleaved fashion.
 * Hence, you can use modulo noMaps to extract the outputs of a single map.
 */
// TODO: This has to be checked.
final class ConvolutionFilter_JVM_Breeze_SparseMM(override val builder:        ConvolutionFilterBuilder,
                                                  override val inputHints:     BuildHints,
                                                  override val seed:           InstanceSeed,
                                                  override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends ConvolutionFilter_JVM_Breeze {
  require(builder        != null)
  require(inputHints     != null)
  require(seed           != null)
  require(weightBufferBuilder != null)


  // ---------------------------------------------------------------------------
  //    Weights related
  // ---------------------------------------------------------------------------
  /**
   * Expanded version of weights that makes computations faster most equations easier.
   */
  // It would be so nice, if we could do this without extra memory. However,
  // the current breeze implementation does not use array offsets.
  // (Hint: In this case, the iterators are the problem.. Fix if possible!)
  protected def generateWEx(inputSize: Size, outputSize: Size)
  : CSCMatrix[Real] = {
    val noInputs  = inputSize.noValues
    val noOutputs = outputSize.noValues
    val data = {
      val res = DenseMatrix.zeros[Real](filter.layout.noValues, noOutputs / noMaps)
      res(::, *) := new DenseVector(filter.values)
      res.data
    }
    val rowIndices = {
      val res = DenseMatrix.zeros[Int](w.rows, noOutputs)
      val tmp = DenseVector.zeros[Int](inputSize.noChannels)
      kernel.foreachValidPair(inputSize, noMaps, (i0, i1, offset0) => {
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
    new CSCMatrix(data, noInputs, noOutputs, colPointers, rowIndices)
  }
  /*
  protected val wExData: DVec = DVec.zeros(kernel.noValues * noOutputs)
  protected val wEx: SMat = {
    val colPointers = (0 to noOutputs).map(i => /*wFlat.offset +*/ i * kernel.noValues).toArray
    val rowIndices = {
      val tmp = Array.ofDim[Int](noOutputs * kernel.noValues)
      kernel.foreachValidPair((i, offset) => {
        val n0 = i * noWeightsPerGroup
        (j, offset) => {
          var w = n0 + j
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
    new CSCMatrix(wExData.data, noInputs, noOutputs, colPointers, rowIndices)
  }
  update() // Do not remove!
  */

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(input:  RealArrayTensor,
                                   output: RealArrayTensor)
  : PredictContext = {
    val inpSize = input.layout.size
    val inp     = input.valuesMatrix
    val outSize = output.layout.size
    val out     = output.valuesMatrix
    val wEx     = generateWEx(inpSize, outSize)
    val wEx_t   = wEx.t
    out := wEx_t * inp
    ConvolutionFilter_JVM_Breeze_SparseMM_Context(wEx)
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(context: PredictContext,
                                            oldError: RealArrayTensor,
                                            newError: RealArrayTensor)
  : Unit = context match {
    case ConvolutionFilter_JVM_Breeze_SparseMM_Context(wEx) =>
      val oldErr = oldError.valuesMatrix
      val newErr = newError.valuesMatrix
      newErr := wEx * oldErr
    case _ =>
      throw new MatchError(context)
  }

}

final case class ConvolutionFilter_JVM_Breeze_SparseMM_Context(wEx: CSCMatrix[Real])
  extends PredictContext {
}

object ConvolutionFilter_JVM_Breeze_SparseMM_Description
  extends ModuleVariant_JVM_Description[ConvolutionFilterBuilder] {

  override def build(builder:       ConvolutionFilterBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : ConvolutionFilter_JVM_Breeze_SparseMM = {
    new ConvolutionFilter_JVM_Breeze_SparseMM(
      builder, hints, seed, weightsBuilder
    )
  }

}
