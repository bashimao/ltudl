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

import breeze.linalg.{DenseMatrix, DenseVector}
import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.sizes._
import org.scalatest._
import TestUtils._
import edu.latrobe.blaze.initializers.GaussianDistributionBuilder
import edu.latrobe.blaze.modules.jvm.{Softmax_JVM_ApacheCommons_Description, Softmax_JVM_Baseline_Description}
import edu.latrobe.cublaze.CUBlaze
import edu.latrobe.cublaze.modules._

final class TestModule_ClassNLLConstraint
  extends FlatSpec with Matchers {

  val epsilon
  : Real = Real(1e-7f)

  "ClassNLL & ClassLL" should "behave the same" in {

    val x = RealArrayTensor.fill(Size1(1, 2), 4, PseudoRNG(4711).gaussianDistribution())
    val y = RealArrayTensor(
      Size1(1, 5), 4,
      Array(
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 1.0f
      )
    )
    val b = Batch(x, y)

    val mb0 = SequenceBuilder(
      LinearFilterBuilder(5),
      SoftmaxBuilder(),
      ClassLLConstraintBuilder()
    )

    val mb1 = SequenceBuilder(
      LinearFilterBuilder(5),
      LogSoftmaxBuilder(),
      ClassNLLConstraintBuilder()
    )

    val ib = GaussianDistributionBuilder().setSeed(BuilderSeed.reproducible())

    val m0 = mb0.build(BuildHints.derive(x))
    m0.reset(ib.build())

    val m1 = mb1.build(BuildHints.derive(x))
    m1.reset(ib.build())

    // Do prediction.
    val yHat0 = m0.predict(Training(0L), b)
    val yHat1 = m1.predict(Training(0L), b)

    // Get gradients.
    val g0 = m0.weightBuffer.allocateZeroedSibling()
    m0.deriveGradients(yHat0, g0)

    val g1 = m1.weightBuffer.allocateZeroedSibling()
    m1.deriveGradients(yHat1, g1)

    val gDiff  = g1 - g0
    val gDiffX = gDiff.banks.head._2.segments.head._2.values

    all(gDiffX) should be < epsilon

    val dummy = 0
  }

}
