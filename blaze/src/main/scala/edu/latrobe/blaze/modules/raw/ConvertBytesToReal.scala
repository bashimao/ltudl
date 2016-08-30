/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2015 Matthias Langer (t3l@threelights.de)
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

package edu.latrobe.blaze.modules.raw

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.sizes._
import scala.util.hashing._

/**
 * Reinterprets the stored bytes as float values.
 */
final class ConvertBytesToReal(override val builder:        ConvertBytesToRealBuilder,
                               override val inputHints:     BuildHints,
                               override val seed:           InstanceSeed,
                               override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends RawLayer[ConvertBytesToRealBuilder]
    with NonTrainableLayer[ConvertBytesToRealBuilder]
    with NonPenalizing {

  val factor
  : Real = Real.one / 0xFF

  override protected def doPredict(input: RawTensor)
  : RealArrayTensor = {
    val inp = input.bytes
    val out = RealArrayTensor.zeros(input.layout)

    out.foreachSamplePair(
      (i, off0, length) => {
        val inpValues = inp(i)
        assume(inpValues.length == input.layout.size.noValues)
        ArrayEx.fill(
          out.values, off0, 1,
          inpValues,  0,    1,
          inpValues.length
        )(MathMacros.toUnsigned(_) * factor)
      }
    )

    out
  }

}

final class ConvertBytesToRealBuilder
  extends RawLayerBuilder[ConvertBytesToRealBuilder] {

  override def repr
  : ConvertBytesToRealBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ConvertBytesToRealBuilder]

  override protected def doCopy()
  : ConvertBytesToRealBuilder = ConvertBytesToRealBuilder()


  // ---------------------------------------------------------------------------
  //   Weights / Building related.
  // ---------------------------------------------------------------------------
  override def weightLayoutFor(hints:   BuildHints,
                               builder: TensorLayoutBufferBuilder)
  : BuildHints = outputHintsFor(hints)

  override def outputHintsFor(hints: BuildHints)
  : BuildHints = hints.derive(JVM, hints.layout)

  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : ConvertBytesToReal = new ConvertBytesToReal(
    this, hints, seed, weightsBuilder
  )

}

object ConvertBytesToRealBuilder {

  final def apply()
  : ConvertBytesToRealBuilder = new ConvertBytesToRealBuilder

}