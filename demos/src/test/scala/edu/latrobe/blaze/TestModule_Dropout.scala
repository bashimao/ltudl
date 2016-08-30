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

package edu.latrobe.blaze

import edu.latrobe._
import edu.latrobe.blaze.initializers._
import edu.latrobe.blaze.gradientchecks._
import edu.latrobe.blaze.modules._
import edu.latrobe.sizes._
import org.scalatest._
import TestUtils._
import breeze.linalg.DenseMatrix

final class TestModule_Dropout
  extends FlatSpec
    with Matchers {

  val x = RealArrayTensor(Size1(1, 3), 3, Array(
    1.0f, 2.0f, 3.0f, // sample 0
    4.0f, 5.0f, 6.0f, // sample 1
    7.0f, 8.0f, 9.0f  // sample 2
  ))

  val initializer0 = UniformDistributionBuilder().setSeed(BuilderSeed.reproducible())
  val initializer1 = UniformDistributionBuilder().setSeed(BuilderSeed.reproducible())

  val y = RealArrayTensor.derive(Size1(1, 2), DenseMatrix.ones[Real](2, 3))

  val e = RealArrayTensor.derive(Size1(1, 2), DenseMatrix.ones[Real](2, 3))

  "Dropout" should "not have any influence when Test mode is used" in {

    // Build 2 sequences. One with dropout and one without dropout.
    val seq0 = SequenceBuilder(
      LinearFilterBuilder(2),
      AddBiasBuilder(),
      //MultiplyValuesBuilder(OperationScope.Batch, Real.pointFive),
      SquaredErrorConstraintBuilder()
    ).build(BuildHints.derive(x))
    seq0.reset(initializer0.build())

    val seq1 = SequenceBuilder(
      LinearFilterBuilder(2),
      AddBiasBuilder(),
      DropoutBuilder(),
      SquaredErrorConstraintBuilder()
    ).build(BuildHints.derive(x))
    seq1.reset(initializer1.build())

    // Run predictions.
    val p0 = seq0.predict(Inference(), x, y)
    val p1 = seq1.predict(Inference(), x, y)

    similarity(p0.output, p1.output) should be < tolerance0
  }

  it should "should filter train-mode gradients correctly" in {

    val rng = PseudoRNG.default

    for (i <- 1 to 10) {
      val w = 1 + rng.nextInt(9)
      val h = 1 + rng.nextInt(9)
      val c = 1 + rng.nextInt(5)
      val n = 128
      val input  = RealArrayTensor.fill(Size2(w, h, c), n, PseudoRNG.default.uniformDistribution())
      val output = RealArrayTensor.ones(Size1(1, 2), n)

      val seq = SequenceBuilder(
        ReshapeBuilder(s => Size1(1, s.noValues)),
        LinearFilterBuilder(output.layout.size.noValues),
        AddBiasBuilder(),
        DropoutBuilder(),
        LinearFilterBuilder(output.layout.size.noValues),
        AddBiasBuilder(),
        SquaredErrorConstraintBuilder()
      ).build(BuildHints.derive(input))
      seq.reset(initializer0.build())

      // Check backward path
      val res = ProbeEntireBufferBuilder().build()(0L, seq, input, output)
      println(f"${input.layout.size} => Rating: ${res.rating}%.4g")
      res.rating should be < tolerance2
    }
  }


}
