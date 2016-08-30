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

package edu.latrobe.blaze

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.blaze.modules.generic.SReLU_Generic_Baseline_Description
import edu.latrobe.blaze.modules.jvm.SReLU_JVM_Baseline_Description
import edu.latrobe.sizes._
import org.scalatest._

final class TestModule_SReLU extends FlatSpec with Matchers {

  "All variants" should "behave as exactly as defined" in {
    val t = 0.7f
    val x = RealArrayTensor(Size1(2, 1), 2, Array(0.0f, 1.0f, -1.0f, 15.0f))
    val y = RealArrayTensor(Size1(2, 1), 2, Array(   t, 1.0f,     t, 15.0f))
    val d = RealArrayTensor(Size1(2, 1), 2, Array(0.0f, 1.0f,  0.0f,  1.0f))
    val e = RealArrayTensor(Size1(2, 1), 2, Array(1.0f, 1.0f,  1.0f,  1.0f))

    // Isolate Baseline variant.
    if (true) {
      SReLUBuilder.unregisterAll()
      SReLUBuilder.register(10, SReLU_JVM_Baseline_Description)
      val layer = SReLUBuilder(t).build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      y shouldEqual prediction.output
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      d shouldEqual dHat
    }

    if (true) {
      SReLUBuilder.unregisterAll()
      SReLUBuilder.register(10, SReLU_Generic_Baseline_Description)
      val layer = SReLUBuilder(t).build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      y shouldEqual prediction.output
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      d shouldEqual dHat
    }

  }

  it should "be able to derive gradients from output" in {
    val t = 0.7f
    val x = RealArrayTensor(Size1(2, 1), 2, Array(0.0f, 1.0f, -1.0f, 15.0f))
    val y = RealArrayTensor(Size1(2, 1), 2, Array(   t, 1.0f,     t, 15.0f))
    val d = RealArrayTensor(Size1(2, 1), 2, Array(0.0f, 1.0f,  0.0f,  1.0f))
    val e = RealArrayTensor(Size1(2, 1), 2, Array(1.0f, 1.0f,  1.0f,  1.0f))

    // Isolate Baseline variant.
    if (true) {
      SReLUBuilder.unregisterAll()
      SReLUBuilder.register(10, SReLU_JVM_Baseline_Description)
      val layer = SequenceBuilder(
        CopyBuilder(),
        SReLUBuilder(t)
      ).build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      y shouldEqual prediction.output
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      d shouldEqual dHat
    }

    if (true) {
      SReLUBuilder.unregisterAll()
      SReLUBuilder.register(10, SReLU_Generic_Baseline_Description)
      val layer = SequenceBuilder(
        CopyBuilder(),
        SReLUBuilder(t)
      ).build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      y shouldEqual prediction.output
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      d shouldEqual dHat
    }

  }

}
