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
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.blaze.modules.jvm._
import edu.latrobe.sizes._
import edu.latrobe.cublaze._
import edu.latrobe.cublaze.modules._
import org.scalatest._

final class TestModule_ReLU extends FlatSpec with Matchers {

  "All variants" should "behave as exactly as defined" in {
    val x = RealArrayTensor(Size1(2, 1), 2, Array(1.0f,  1.0f, -1.0f, 15.0f))
    val y = RealArrayTensor(Size1(2, 1), 2, Array(1.0f,  1.0f,  0.0f, 15.0f))
    val d = RealArrayTensor(Size1(2, 1), 2, Array(2.0f, -2.0f,  0.0f,  1.0f))
    val e = RealArrayTensor(Size1(2, 1), 2, Array(2.0f, -2.0f,  1.0f,  1.0f))

    // Isolate Baseline variant.
    if (true) {
      ReLUBuilder.unregisterAll()
      ReLUBuilder.register(10, ReLU_JVM_Baseline_Description)
      val layer = ReLUBuilder().build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      y shouldEqual prediction.output
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      d shouldEqual dHat
    }

    // Isolate CUDA Direct variant.
    if (true) {
      ReLUBuilder.unregisterAll()
      ReLUBuilder.register(10, ReLU_CUDA_CUDNN_Description)
      val layer = ReLUBuilder().build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      y shouldEqual prediction.output
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      d shouldEqual dHat
    }
  }

  it should "be able to derive gradients from output" in {
    val x = RealArrayTensor(Size1(2, 1), 2, Array(1.0f,  1.0f, -1.0f, 15.0f))
    val y = RealArrayTensor(Size1(2, 1), 2, Array(1.0f,  1.0f,  0.0f, 15.0f))
    val d = RealArrayTensor(Size1(2, 1), 2, Array(2.0f, -2.0f,  0.0f,  1.0f))
    val e = RealArrayTensor(Size1(2, 1), 2, Array(2.0f, -2.0f,  1.0f,  1.0f))

    // Isolate Baseline variant.
    if (true) {
      ReLUBuilder.unregisterAll()
      ReLUBuilder.register(10, ReLU_JVM_Baseline_Description)
      val layer = SequenceBuilder(
        CopyBuilder(),
        ReLUBuilder()
      ).build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      y shouldEqual prediction.output
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      d shouldEqual dHat
    }

    // Isolate CUDA Direct variant.
    if (true) {
      ReLUBuilder.unregisterAll()
      ReLUBuilder.register(10, ReLU_CUDA_CUDNN_Description)
      val layer = SequenceBuilder(
        CopyBuilder(),
        ReLUBuilder()
      ).build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      y shouldEqual prediction.output
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      d shouldEqual dHat
    }

  }

}
