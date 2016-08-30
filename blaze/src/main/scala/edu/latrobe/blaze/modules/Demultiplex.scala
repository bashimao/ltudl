/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2014 Matthias Langer (t3l@threelights.de)
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
 */

package edu.latrobe.blaze.modules

import breeze.linalg.{DenseMatrix, DenseVector}
import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.TensorDependency._

/**
 *  Be careful! This one cannot be concatenated!
 */
final class Demultiplex(override val builder:        DemultiplexBuilder,
                        override val inputHints:     BuildHints,
                        override val seed:           InstanceSeed,
                        override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends Layer[DemultiplexBuilder]
    with NonTrainableLayer[DemultiplexBuilder]
    with NonPenalizing {

  override val outputHints: BuildHints = builder.outputHintsFor(inputHints)


  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  // TODO: Use better demultiplex code already in "ltu"!
  def demultiplex(src:        DenseVector[Real],
                  dst:        DenseVector[Real],
                  noChannels: Int)
  : Unit = {
    val channelSize = src.length / noChannels
    var dst0 = 0
    var c    = 0
    while (c < noChannels) {
      val dst1 = dst0 + channelSize
      dst(dst0 until dst1) := src(c until src.length by noChannels)
      dst0 = dst1
      c += 1
    }
  }

  def multiplex(src:        DenseVector[Real],
                dst:        DenseVector[Real],
                noChannels: Int)
  : Unit = {
    val channelSize = src.length / noChannels
    var src0 = 0
    var c    = 0
    while (c < noChannels) {
      val src1 = src0 + channelSize
      dst(c until src.length by noChannels) := src(src0 until src1)
      src0 = src1
      c += 1
    }
  }

  // TODO: Use better function!
  def processBatch(src: DenseMatrix[Real],
                   fn:  (DenseVector[Real], DenseVector[Real]) => Unit)
  : DenseMatrix[Real] = {
    val dst = DenseMatrix.zeros[Real](src.rows, src.cols)
    var i = 0
    while (i < src.cols) {
      fn(src(::, i), dst(::, i))
      i += 1
    }
    dst
  }

  override protected def doPredict(mode:           Mode,
                                   inPlaceAllowed: Boolean,
                                   input:          Tensor,
                                   reference:      Tensor)
  : (Tensor, PredictContext) = {
    val inp     = input.valuesMatrix
    val inpSize = input.layout.size
    val outSize = inpSize.demultiplex
    val out     = processBatch(inp, demultiplex(_, _, inpSize.noChannels))
    val output  = RealArrayTensor.derive(outSize, out)
    (output, EmptyContext)
  }

  override protected def doPredictInv(output: Tensor, context: PredictContext)
  : Tensor = {
    val out     = output.valuesMatrix
    val inpSize = output.layout.size.multiplex
    val inp     = processBatch(out, multiplex(_, _, inpSize.noChannels))
    RealArrayTensor.derive(inpSize, inp)
  }


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.NotRequired

  override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.NotRequired

  override protected def doDeriveInputError(input:     Tensor,
                                            reference: Tensor,
                                            output:    Tensor,
                                            context:   PredictContext,
                                            error:     Tensor)
  : Tensor = {
    val err     = error.valuesMatrix
    val errSize = error.layout.size.multiplex
    processBatch(err, multiplex(_, _, errSize.noChannels))
    RealArrayTensor.derive(errSize, err)
  }

}

final class DemultiplexBuilder
  extends LayerBuilder[DemultiplexBuilder]
    with NonTrainableLayerBuilder[DemultiplexBuilder] {

  override def repr: DemultiplexBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[DemultiplexBuilder]

  override protected def doCopy()
  : DemultiplexBuilder = DemultiplexBuilder()


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def weightLayoutFor(hints:   BuildHints,
                               builder: TensorLayoutBufferBuilder)
  : BuildHints = outputHintsFor(hints)

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = hints.derive(
    JVM,
    hints.layout.derive(hints.layout.size.demultiplex)
  )

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Demultiplex = new Demultiplex(this, hints, seed, weightsBuilder)

}

object DemultiplexBuilder {

  final def apply()
  : DemultiplexBuilder = new DemultiplexBuilder

}
