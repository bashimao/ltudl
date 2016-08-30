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

// TODO: This implementation is somewhat inefficient if we do not save the index of the winning activation during predict. On the other hand, this will require additional memory. We should benchmark this!
final class MaxPooling_JVM_Baseline(override val builder:        MaxPoolingBuilder,
                                    override val inputHints:     BuildHints,
                                    override val seed:           InstanceSeed,
                                    override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends MaxPooling_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(mode:   Mode,
                                   input:  RealArrayTensor,
                                   output: RealArrayTensor)
  : PredictContext = {
    output := Real.negativeInfinity
    val inp       = input.valuesMatrix
    val inpLayout = input.layout
    val inpSize   = inpLayout.size
    val out       = output.valuesMatrix

    mode match {
      case mode: Training =>
        val indices = Array.ofDim[Int](out.cols, out.rows)
        kernel.foreachValidPair(inpSize, inpSize.noChannels,
          (i0, i1, offset0) => {
            val dst = out(i0 until i1, ::)

            (j0, j1, offset0, offset1) => {
              val src = inp(offset0 until offset1, ::)
              MatrixEx.transformPairs(dst, src)((row, col, dst, src) => {
                if (src > dst) {
                  indices(col)(i0 + row) = offset0 + row
                  src
                }
                else {
                  dst
                }
              })
            }
          }
        )
        MaxPooling_JVM_Baseline_Context(inpLayout, indices)

      case mode: Inference =>
        kernel.foreachValidPair(inpSize, inpSize.noChannels,
          (i0, i1, offset0) => {
            val dst = out(i0 until i1, ::)

            (j0, j1, offset0, offset1) => {
              val src = inp(offset0 until offset1, ::)
              MatrixEx.transform(dst, src)(
                (dst, src) => if (src > dst) src else dst
              )
            }
          }
        )
        EmptyContext

      case _ =>
        throw new MatchError(mode)
    }
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  /**
   * Invert max-selection process by finding the responsible activation for each
   * neuron.
   */
  override protected def doDeriveInputError(context:  PredictContext,
                                            oldError: RealArrayTensor,
                                            newError: RealArrayTensor)
  : Unit = context match {
    case MaxPooling_JVM_Baseline_Context(inpLayout, indices) =>
      val oldErr = oldError.valuesMatrix
      val newErr = newError.valuesMatrix

      val offset0 = newErr.offset
      val stride  = newErr.majorStride
      MatrixEx.foreachPair(oldErr)(
        (outRow, col, v) => {
          val inpRow = indices(col)(outRow)
          val offset = offset0 + inpRow + col * stride
          newErr.data(offset) += v
        }
      )

      /*
      The previous method, which does not handle ties correctly.
      kernel.foreachValidPair(input.size, input.size.noChannels,
        (i0, i1, offset0) => {
          val srcRange = i0 until i1
          val srcErr = oldErr(srcRange, ::)
          val srcVal = out(srcRange, ::)

          (j0, j1, offset0, offset1) => {
            val dstRange = offset0 until offset1
            val dstErr = newErr(dstRange, ::)
            val dstVal = inp(dstRange, ::)

            // TODO: Can avoid using destination value buffer here. Actually, it would be much better if we'd use use an index to avoid this lookup completely.
            dstErr.transformEx(dstVal, srcErr, srcVal)(
              (de, dv, se, sv) => if (dv == sv) de + se else de
            )
          }
        }
      )
      */
    case _ =>
      throw new MatchError(context)
  }

}

object MaxPooling_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[MaxPoolingBuilder] {

  override def build(builder:        MaxPoolingBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : MaxPooling_JVM_Baseline = new MaxPooling_JVM_Baseline(
    builder, hints, seed, weightsBuilder
  )

}

final case class MaxPooling_JVM_Baseline_Context(override val inputLayout: IndependentTensorLayout,
                                                 rowIndices:               Array[Array[Int]])
  extends MaxPooling_JVM_Context {
  require(inputLayout != null && rowIndices != null)
}
