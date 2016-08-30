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
import edu.latrobe.sizes._
import org.scalatest._
import TestUtils._
import edu.latrobe.blaze.modules.jvm.{Sigmoid_JVM_ApacheCommons_Description, Sigmoid_JVM_Baseline_Description}
import edu.latrobe.cublaze.modules._

final class TestModule_Sigmoid extends FlatSpec with Matchers {

  def sigmoid(c: Real, x: Real): Real = Real(1.0 / (1.0 + Math.exp(-c * x)))

  def sigmoid_dx(c: Real, x: Real)
  : Real = c * (Real.one - sigmoid(c, x)) * sigmoid(c, x)

  "All Sigmoid variants" should "behave as exactly as defined" in {
    val x = RealArrayTensor(Size1(2, 1), 2, Array(0.0f, 1.0f, -1.0f, 15.0f))
    val y = RealArrayTensor(Size1(2, 1), 2, ArrayEx.map(x.values)(sigmoid(Real.one, _)))
    val d = RealArrayTensor(Size1(2, 1), 2, ArrayEx.map(x.values)(sigmoid_dx(Real.one, _)))
    val e = RealArrayTensor(Size1(2, 1), 2, Array(1.0f, 1.0f,  1.0f,  1.0f))

    // Isolate Baseline variant.
    if (true) {
      SigmoidBuilder.unregisterAll()
      SigmoidBuilder.register(10, Sigmoid_JVM_Baseline_Description)
      val layer = SigmoidBuilder().build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      similarity(y, prediction.output) should be < tolerance0
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      similarity(d, dHat) should be < tolerance0
    }

    // Isolate Apache Commons variant.
    if (true) {
      SigmoidBuilder.unregisterAll()
      SigmoidBuilder.register(10, Sigmoid_JVM_ApacheCommons_Description)
      val layer = SigmoidBuilder().build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      similarity(y, prediction.output) should be < tolerance0
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      similarity(d, dHat) should be < tolerance0
    }

    // Isolate CUDA Sandbox variant.
    /*
    if (true) {
      SigmoidBuilder.unregisterAll()
      SigmoidBuilder.register(Sigmoid_CUDA_CUDNN_Sandbox_Description)
      val layer = SigmoidBuilder().build(BuildHints.derive(Train, x))
      val prediction = layer.predict(x, null)
      similarity(y, prediction.output) should be < tolerance0
      val dHat = layer.deriveInputError(prediction, e).get
      similarity(d, dHat) should be < tolerance0
    }
    */

    // Isolate CUDA Direct variant.
    if (true) {
      SigmoidBuilder.unregisterAll()
      SigmoidBuilder.register(10, Sigmoid_CUDA_CUDNN_Description)
      val layer = SigmoidBuilder().build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      similarity(y, prediction.output) should be < tolerance0
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      similarity(d, dHat) should be < tolerance0
    }
  }

  /*
  "The CPU variants" should "be able to handle scale factors in" in {
    val c: Real = 0.7f
    val x = Tensor.derive(Size1(2, 1), DMat(2, 2, 0.0f, 1.0f, -1.0f, 15.0f))
    val y = x.mapValues(sigmoid(c, _))
    val d = x.mapValues(sigmoid_dx(c, _))
    val e = Tensor.derive(Size1(2, 1), DMat(2, 2, 1.0f, 1.0f,  1.0f,  1.0f))

    // Isolate Baseline variant.
    if (true) {
      SigmoidBuilder.unregisterAll()
      SigmoidBuilder.register(Sigmoid_JVM_Baseline_Description)
      val layer = SigmoidBuilder(c).build(BuildHints.derive(Train, x))
      val prediction = layer.predict(x, null)
      similarity(y, prediction.output) should be < tolerance0
      val dHat = layer.deriveInputError(prediction, e).get
      similarity(d, dHat) should be < tolerance0
    }

    // Isolate ApacheCommons variant.
    if (true) {
      SigmoidBuilder.unregisterAll()
      SigmoidBuilder.register(Sigmoid_JVM_ApacheCommons_Description)
      val layer = SigmoidBuilder(c).build(BuildHints.derive(Train, x))
      val prediction = layer.predict(x, null)
      similarity(y, prediction.output) should be < tolerance0
      val dHat = layer.deriveInputError(prediction, e).get
      similarity(d, dHat) should be < tolerance0
    }
  }

  "The baseline variant" should "be able to handle variable precision" in {
    val c: Real = 0.7f
    val x = Tensor.derive(Size1(2, 1), DMat(2, 2, 0.0f, 1.0f, -1.0f, 15.0f))
    val y = x.mapValues(sigmoid(c, _))
    val d = x.mapValues(sigmoid_dx(c, _))
    val e = Tensor.derive(Size1(2, 1), DMat(2, 2, 1.0f, 1.0f,  1.0f,  1.0f))

    // Isolate Baseline variant.
    for (precision <- Array(4, 8, 12, 16, 24, 32, 48)) {
      SigmoidBuilder.unregisterAll()
      SigmoidBuilder.register(Sigmoid_JVM_Baseline_Description)
      val layer = SigmoidBuilder(c, precision).build(BuildHints.derive(Train, x))
      val prediction = layer.predict(x, null)
      similarity(y, prediction.output) should be < tolerance0
      val dHat = layer.deriveInputError(prediction, e).get
      similarity(d, dHat) should be < tolerance0
    }
  }
  */

}
