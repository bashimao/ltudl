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

final class MeanPooling_JVM_Baseline(override val builder:        MeanPoolingBuilder,
                                     override val inputHints:     BuildHints,
                                     override val seed:           InstanceSeed,
                                     override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends MeanPooling_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredictForTraining(input: RealArrayTensor,
                                              output: RealArrayTensor)
  : Array[Real] = {
    val inp        = input.valuesMatrix
    val inpLayout  = input.layout
    val inpSize    = inpLayout.size
    val noChannels = inpSize.noChannels
    val out        = output.valuesMatrix

    if (includePadding) {
      val alpha = Real.one / kernel.noValues
      kernel.foreachValidPair(inpSize, noChannels,
        (i0, i1, offset0) => {
          val dst = out(i0 until i1, ::)

          (j0, j1, offset0, offset1) => {
            val src = inp(offset0 until offset1, ::)
            MatrixEx.add(
              dst, alpha, src
            )
          }
        }
      )
      Array(alpha)
    }
    else {
      val counts     = new Array[Int](output.layout.size.noTuples)
      var countIndex = 0
      kernel.foreachValidPairEx(inpSize, noChannels,
        (i0, i1, offset0) => {
          val dst = out(i0 until i1, ::)

          var n = 0
          Tuple2(
            (j0, j1, offset0, offset1) => {
              val src = inp(offset0 until offset1, ::)
              n += 1
              val w = Real.one / n
              MatrixEx.lerp(dst, src, w)
            },
            () => {
              counts(countIndex) = n
              countIndex += 1
            }
          )
        }
      )
      ArrayEx.map(counts)(Real.one / _)
    }
  }

  override protected def doPredictForInference(input: RealArrayTensor,
                                               output: RealArrayTensor)
  : Unit = {
    val inp        = input.valuesMatrix
    val inpLayout  = input.layout
    val inpSize    = inpLayout.size
    val noChannels = inpSize.noChannels
    val out        = output.valuesMatrix

    if (includePadding) {
      val alpha = Real.one / kernel.noValues
      kernel.foreachValidPair(inpSize, noChannels,
        (i0, i1, offset0) => {
          val dst = out(i0 until i1, ::)

          (j0, j1, offset0, offset1) => {
            val src = inp(offset0 until offset1, ::)
            MatrixEx.add(dst, alpha, src)
          }
        }
      )
    }
    else {
      kernel.foreachValidPair(inpSize, noChannels,
        (i0, i1, offset0) => {
          val dst = out(i0 until i1, ::)

          var n = 0
          (j0, j1, offset0, offset1) => {
            val src = inp(offset0 until offset1, ::)
            n += 1
            val w = Real.one / n
            MatrixEx.lerp(dst, src, w)
          }
        }
      )
    }
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  /**
   * Since we average on predict, we have errors regarding the averages of many
   * activations. Hence, we simply reverse the averaging process here.
   */
  override protected def doDeriveInputError(countsInv: Array[Real],
                                            oldError:  RealArrayTensor,
                                            newError:  RealArrayTensor)
  : Unit = {
    val oldErr  = oldError.valuesMatrix
    val newErr  = newError.valuesMatrix
    val inpSize = newError.layout.size

    if (includePadding) {
      val alpha = countsInv(0)

      // Scale down output error.
      oldErr *= alpha

      // Up-sample error.
      kernel.foreachValidPair(inpSize, inpSize.noChannels,
        (i0, i1, offset0) => {
          val src = oldErr(i0 until i1, ::)

          (j0, j1, offset0, offset1) => {
            val dst = newErr(offset0 until offset1, ::)
            dst += src
          }
        }
      )
    }
    else {
      // Up-sample error.
      var countIndex = 0
      kernel.foreachValidPair(inpSize, inpSize.noChannels,
        (i0, i1, offset0) => {
          // Scale down output error.
          val src = oldErr(i0 until i1, ::)
          // TODO: Could do this in one go by just reshaping the error.
          src *= countsInv(countIndex)
          countIndex += 1

          (j0, j1, offset0, offset1) => {
            val dst = newErr(offset0 until offset1, ::)
            dst += src
          }
        }
      )
    }
  }

}

object MeanPooling_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[MeanPoolingBuilder] {

  override def build(builder:        MeanPoolingBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : MeanPooling_JVM_Baseline = new MeanPooling_JVM_Baseline(
    builder, hints, seed, weightsBuilder
  )

}
