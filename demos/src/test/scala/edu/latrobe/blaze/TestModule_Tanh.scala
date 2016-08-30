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
import edu.latrobe.blaze.modules._
import edu.latrobe.sizes._
import org.scalatest._
import TestUtils._
import edu.latrobe.blaze.modules.jvm._
import edu.latrobe.cublaze.modules._

final class TestModule_Tanh extends FlatSpec with Matchers {

  def tanh(x: Real): Real = Real(Math.tanh(x))

  def tanh_dx(x: Real): Real = Real.one - tanh(x) * tanh(x)

  "All variants" should "behave as exactly as defined" in {
    val x = RealArrayTensor(Size1(2, 1), 2, Array(0.0f, 1.0f, -1.0f, 15.0f))
    val y = RealArrayTensor(Size1(2, 1), 2, ArrayEx.map(x.values)(tanh))
    val d = RealArrayTensor(Size1(2, 1), 2, ArrayEx.map(x.values)(tanh_dx))
    val e = RealArrayTensor(Size1(2, 1), 2, Array(1.0f, 1.0f,  1.0f,  1.0f))

    // Isolate Baseline variant.
    if (true) {
      TanhBuilder.unregisterAll()
      TanhBuilder.register(10, Tanh_JVM_Baseline_Description)
      val layer = TanhBuilder().build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      similarity(y, prediction.output) should be < tolerance0
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      similarity(d, dHat) should be < tolerance0
    }

    // Isolate Apache Commons variant.
    if (true) {
      TanhBuilder.unregisterAll()
      TanhBuilder.register(10, Tanh_JVM_ApacheCommons_Description)
      val layer = TanhBuilder().build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      similarity(y, prediction.output) should be < tolerance0
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      similarity(d, dHat) should be < tolerance0
    }

    // Isolate CUDA Sandbox variant.
    /*
    if (true) {
      TanhBuilder.unregisterAll()
      TanhBuilder.register(Tanh_CUDA_CUDNN_Sandbox_Description)
      val layer = TanhBuilder().build(BuildHints.derive(Train, x))
      val prediction = layer.predict(x, null)
      similarity(y, prediction.output) should be < tolerance0
      val dHat = layer.deriveInputError(prediction, e).get
      similarity(d, dHat) should be < tolerance0
    }
    */

    // Isolate CUDA Sandbox variant.
    if (true) {
      TanhBuilder.unregisterAll()
      TanhBuilder.register(10, Tanh_CUDA_CUDNN_Description)
      val layer = TanhBuilder().build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      similarity(y, prediction.output) should be < tolerance0
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      similarity(d, dHat) should be < tolerance0
    }
  }

}
