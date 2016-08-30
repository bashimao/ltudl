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

/**
  * This is just a direct implementation of Sqrt(MeanPool(Sqr(x)) + e)
  */
final class L2Pooling_JVM_Baseline(override val builder:        L2PoolingBuilder,
                                   override val inputHints:     BuildHints,
                                   override val seed:           InstanceSeed,
                                   override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends L2Pooling_JVM {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredict(mode:   Mode,
                                   input:  RealArrayTensor,
                                   output: RealArrayTensor)
  : PredictContext = {
    val inp        = input.valuesMatrix
    val inpLayout  = input.layout
    val inpSize    = inpLayout.size
    val noChannels = inpSize.noChannels
    val out        = output.valuesMatrix

    val context = {
      if (includePadding) {
        val alpha = Real.one / kernel.noValues
        // TODO: Create alternative method that does not use a temporary buffer.
        kernel.foreachValidPair(inpSize, noChannels,
          (i0, i1, offset0) => {
            val dst = out(i0 until i1, ::)

            (j0, j1, offset0, offset1) => {
              val src = inp(offset0 until offset1, ::)
              MatrixEx.fill(dst, src)(
                x => alpha * x * x
              )
            }
          }
        )
        SizeContext(inpSize)
      }
      else {
        val counts = new Array[Int](output.layout.size.noTuples)
        kernel.foreachValidPairEx(inpSize, noChannels,
          (i0, i1, offset0) => {
            val dst = out(i0 until i1, ::)

            val i = i0 / noChannels
            var n = 0
            Tuple2(
              (j0, j1, offset0, offset1) => {
                val src = inp(offset0 until offset1, ::)
                n += 1
                val w = Real.one / n
                MatrixEx.lerp(dst, src, w)
              },
              () => counts(i) = n
            )
          }
        )

        L2Pooling_JVM_Baseline_Context(
          inpLayout, ArrayEx.map(counts)(Real.one / _)
        )
      }
    }

    // (this is the 1st reason why we added the regularizer)
    MatrixEx.transform(out)(
      x => Real(Math.sqrt(x + epsilon))
    )

    context
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override protected def doDeriveInputError(input:    RealArrayTensor,
                                            output:   RealArrayTensor,
                                            context:  PredictContext,
                                            oldError: RealArrayTensor,
                                            newError: RealArrayTensor)
  : Unit = {
    val inp    = input.valuesMatrix
    val out    = output.valuesMatrix
    val oldErr = oldError.valuesMatrix
    val newErr = newError.valuesMatrix

    // Scale down output error. (this is the 2nd reason why we added the regularizer)
    oldErr :/= out

    context match {
      case SizeContext(inpSize) =>
        oldErr *= Real.one / kernel.noValues

        // Up-sample error.
        kernel.foreachValidPair(inpSize, inpSize.noChannels,
          (i0, i1, offset0) => {
            // Scale down output error.
            val src = oldErr(i0 until i1, ::)

            (j0, j1, offset0, offset1) => {
              val dst = newErr(offset0 until offset1, ::)
              dst += src
            }
          }
        )

      case L2Pooling_JVM_Baseline_Context(inputLayout, countsInv) =>
        val inpSize    = inputLayout.size
        val noChannels = inpSize.noChannels

        // Up-sample error.
        kernel.foreachValidPair(inpSize, noChannels,
          (i0, i1, offset0) => {
            // Scale down output error.
            val src = oldErr(i0 until i1, ::)
            // TODO: Could do this in one go by just reshaping the error.
            src *= countsInv(i0 / noChannels)

            (j0, j1, offset0, offset1) => {
              val dst = newErr(offset0 until offset1, ::)
              dst += src
            }
          }
        )
    }

    // Scale up-sampled error by the respective input.
    newErr :*= inp
  }

}

final case class L2Pooling_JVM_Baseline_Context(inputLayout: IndependentTensorLayout,
                                                countsInv:   Array[Real])
  extends PredictContext {
  require(inputLayout != null && countsInv != null)
}

object L2Pooling_JVM_Baseline_Description
  extends ModuleVariant_JVM_Description[L2PoolingBuilder] {

  override def build(builder:       L2PoolingBuilder,
                     hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : L2Pooling_JVM_Baseline = new L2Pooling_JVM_Baseline(
    builder, hints, seed, weightsBuilder
  )

}
