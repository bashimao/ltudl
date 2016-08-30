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
import edu.latrobe.blaze.modules.UnpoolBuilder

final class Unpool_JVM_Baseline(override val builder:        UnpoolBuilder,
                                override val inputHints:     BuildHints,
                                override val seed:           InstanceSeed,
                                override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Unpool_JVM {

  private lazy val inpCountsInv: Array[Real] = {
    val tmp = new Array[Int](inputSizeHint.noTuples)
    kernel.foreachValidPair(outputSize, 1, (i0, i1, offset0) => {
      (j0, j1, offset0, offset1) => {
        tmp(i0) += 1
      }
    })
    ArrayEx.map(tmp)(Real.one / _)
  }

  private lazy val outCountsInv: Array[Real] = {
    val tmp = new Array[Int](outputSize.noTuples)
    kernel.foreachValidPair(outputSize, 1, (i0, i1, offset0) => {
      (j0, j1, offset0, offset1) => {
        tmp(offset0) += 1
      }
    })
    ArrayEx.map(tmp)(Real.one / _)
  }


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  /*
  override protected def doPredict(mode:    ComputeMode,
                                      output:  SampleTensor,
                                      context: Any)
  : SampleTensor = context match {
    case inpSize: Size =>
      val out = output.values
      val inp = DVec.zeros(inpSize.noValues)
      // TODO: Could be done more efficiently.
      val weights = Array.ofDim[Int](inp.length / inpSize.noChannels)
      kernel.foreachValidPair(inpSize, inpSize.noChannels,
        (i0, i1, offset0) => {
          val src = out(i0 until i1)
          /*val src = {
            val src0 = i * noChannels
            out(src0 until src0 + noChannels)
          }*/
          (j0, j1, offset0, offset1) => {
            /*val dst = {
              val dst0 = offset * noChannels
              inp(dst0 until dst0 + noChannels)
            }*/
            val dst = inp(offset0 until offset1)
            dst += src
            weights(offset0 / inpSize.noChannels) += 1
          }
        }
      )
      val weightsInv = DVec(weights.map(Real.one / _))
      val inpValues = inp.asMatrix(inpSize.noChannels, weightsInv.length)
      inpValues(*, ::) :*= weightsInv// kernelWeightsInv
      DenseSampleTensor(inp, inpSize)
      /*
      // Add items and divide by count.
      val values      = DenseVec.zeros(inputSize)
      val activations = rawOutput.values
      kernel.foreachPairAbs((i, j, offset) => {
        val src0 = j      * noChannels
        val src1 = src0   + noChannels
        val dst0 = offset * noChannels
        val dst1 = dst0   + noChannels
        values(dst0 until dst1) += activations(src0 until src1)
      })
      // Divide by number of values accumulated.
      values :/= outputWeights
      new DenseSampleActivations(values)
      */
  }
  */
  override protected def doPredict(input:  RealArrayTensor,
                                   output: RealArrayTensor)
  : Unit = {
    // TODO: Add faster code that works for full coverage cases.
    val out        = output.valuesMatrix
    val inp        = input.valuesMatrix
    val noChannels = input.layout.size.noChannels

    kernel.foreachValidPair(outputSize, noChannels,
      (i0, i1, offset0) => {
        val src = inp(i0 until i1, ::)

        (j0, j1, offset0, offset1) => {
          val countInv = outCountsInv(offset0 / noChannels)

          // Merge inp with out.
          val dst = out(offset0 until offset1, ::)
          MatrixEx.add(dst, countInv, src)
        }
      }
    )
    /*
    context match {
    case SizeContext(size) =>
      val out = output.valuesMatrix
      val inp = DMat.zeros(size.noTuples, out.cols)
      // TODO: Could be done more efficiently.
      val weights = Array.ofDim[Int](inp.rows / size.noChannels)
      kernel.foreachValidPair(size, size.noChannels,
        (i0, i1, offset0) => {
          /*val src = {
            val src0 = i * noChannels
            out(src0 until src0 + noChannels, ::)
          }*/
          val src = out(i0 until i1, ::)
          (j0, j1, offset0, offset1) => {
            /*
            val dst = {
              val dst0 = offset * noChannels
              inp(dst0 until dst0 + noChannels, ::)
            }
            */
            val dst = inp(offset0 until offset1, ::)
            dst += src
            weights(offset0 / size.noChannels) += 1
          }
        }
      )
      val weightsInv = DVec(weights.map(Real.one / _))
      inp.fastForeachCol(inp => {
        val inpValues = inp.asMatrix(size.noChannels, weightsInv.length)
        inpValues(*, ::) :*= weightsInv
      })
      //val inpValues = inp.reshapeEx(noChannels, weightsInv.length)
      //inpValues(*, ::) :*= kernelWeightsInv
      JVMTensor(size, inp)
    /*
    // Add items and divide by count.
    val values      = DenseMat.zeros(inputSize, rawOutput.noSamples)
    val activations = rawOutput.values
    kernel.foreachPairAbs((i, j, offset) => {
      val src0 = j      * noChannels
      val src1 = src0   + noChannels
      val dst0 = offset * noChannels
      val dst1 = dst0   + noChannels
      values(dst0 until dst1, ::) += activations(src0 until src1, ::)
    })
    // Divide by number of values accumulated.
    values(::, *) :/= outputWeights
    new DenseBatchActivations(values)
   */
   * */
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(oldError: RealArrayTensor,
                                            newError: RealArrayTensor)
  : Unit = {
    val noChannels = oldError.layout.size.noChannels
    val oldErr     = oldError.valuesMatrix
    val newErr     = newError.valuesMatrix

    kernel.foreachValidPair(outputSize, noChannels,
      (i0, i1, offset0) => {
        val dst      = newErr(i0 until i1, ::)
        val countInv = inpCountsInv(i0 / noChannels)

        (j0, j1, offset0, offset1) => {
          // Merge inp with out.
          val src = oldErr(offset0 until offset1, ::)
          MatrixEx.add(dst, countInv, src)
        }
      }
    )
  }

}

object Unpool_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[UnpoolBuilder] {

  override def build(builder:        UnpoolBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Unpool_JVM_Baseline = new Unpool_JVM_Baseline(
    builder, hints, seed, weightsBuilder
  )

}
