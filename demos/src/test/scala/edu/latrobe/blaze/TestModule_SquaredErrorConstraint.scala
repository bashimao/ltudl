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

import breeze.linalg._
import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.sizes._
import org.scalatest._
import TestUtils._

final class TestModule_SquaredErrorConstraint extends FlatSpec with Matchers {

  "All variants" should "behave exactly as defined" in {
    val x = RealArrayTensor(Size1(2, 1), 2, Array( 1.0f, -1.0f, -2.0f, 15.0f))
    val r = RealArrayTensor(Size1(2, 1), 2, Array(15.0f, 15.0f, 15.0f, 15.0f))
    val y = x.copy
    val c = Real(0.5 * sum(((y - r) :* (y - r)).valuesMatrix) / x.layout.noSamples)
    val e = RealArrayTensor(Size1(2, 1), 2, Array(1.0f, 1.0f, 1.0f, 1.0f))
    val d = e + (y - r) * (Real.one / x.layout.noSamples)

    val layer = SquaredErrorConstraintBuilder().setScaleCoefficient(0.5f).build(BuildHints.derive(x))
    val eval = layer.predict(Training(0L), x, r)
    y shouldEqual eval.output
    eval.cost.value shouldEqual c
    val gHat = layer.weightBuffer.allocateZeroedSibling()
    val dHat = layer.deriveGradients(eval, e, gHat).compute()
    similarity(d, dHat) should be < tolerance0
  }

}
