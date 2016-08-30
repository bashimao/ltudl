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

import breeze.numerics._
import breeze.linalg._
import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.sizes._
import org.scalatest._
import TestUtils._
import edu.latrobe.blaze.modules.generic.LogSoftmax_Generic_Baseline_Description
import edu.latrobe.blaze.modules.jvm.LogSoftmax_JVM_Baseline_Description
import edu.latrobe.cublaze.modules._

final class TestModule_LogSoftmax extends FlatSpec with Matchers {

  def softmax(x: DenseVector[Real]): DenseVector[Real] = {
    val y = exp(x)
    y / sum(y)
  }

  def logSoftmax(x: DenseVector[Real]): DenseVector[Real] = log(softmax(x))

  def logSoftmax(x: DenseMatrix[Real]): DenseMatrix[Real] = {
    val result = DenseMatrix.zeros[Real](x.rows, x.cols)
    MatrixEx.foreachColumnVector(result, x)(_ := logSoftmax(_))
    result
  }

  def logSoftmax_dx(x: DenseVector[Real], err: DenseVector[Real])
  : DenseVector[Real] = {
    val y = logSoftmax(x)
    err - exp(y) * sum(err)
  }

  def logSoftmax_dx(x: DenseMatrix[Real], err: DenseMatrix[Real])
  : DenseMatrix[Real] = {
    val result = DenseMatrix.zeros[Real](x.rows, x.cols)
    MatrixEx.foreachColumnVector(result, x, err)(
      _ := logSoftmax_dx(_, _)
    )
    result
  }

  "All LogSoftmax variants" should "behave as exactly as defined" in {

    val x = RealArrayTensor.fill(Size1(3, 1), 4, PseudoRNG.default.gaussianDistribution())
    val y = RealArrayTensor.derive(x.layout.size, logSoftmax(x.valuesMatrix))
    val e = RealArrayTensor.fill(x.layout.size, 4, PseudoRNG.default.gaussianDistribution())
    val d = RealArrayTensor.derive(x.layout.size, logSoftmax_dx(x.valuesMatrix, e.valuesMatrix))

    // Isolate Generic variant.
    if (true) {
      LogSoftmaxBuilder.unregisterAll()
      LogSoftmaxBuilder.register(10, LogSoftmax_Generic_Baseline_Description)
      val layer = LogSoftmaxBuilder().build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      similarity(y, prediction.output) should be < tolerance0
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      similarity(d, dHat) should be < tolerance0
    }

    // Isolate CPU Baseline variant.
    if (true) {
      LogSoftmaxBuilder.unregisterAll()
      LogSoftmaxBuilder.register(10, LogSoftmax_JVM_Baseline_Description)
      val layer = LogSoftmaxBuilder().build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      similarity(y, prediction.output) should be < tolerance0
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      similarity(d, dHat) should be < tolerance0
    }

    // Isolate CUDA Direct variant.
    if (true) {
      LogSoftmaxBuilder.unregisterAll()
      LogSoftmaxBuilder.register(10, LogSoftmax_CUDA_CUDNN_Description)
      val layer = LogSoftmaxBuilder().build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      similarity(y, prediction.output) should be < tolerance0
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      similarity(d, dHat) should be < tolerance0
    }
  }

}
