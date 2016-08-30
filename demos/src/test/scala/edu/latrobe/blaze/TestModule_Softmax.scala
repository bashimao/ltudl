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
import edu.latrobe.blaze.modules.jvm.{Softmax_JVM_ApacheCommons_Description, Softmax_JVM_Baseline_Description}
import edu.latrobe.cublaze.modules._

final class TestModule_Softmax
  extends FlatSpec with Matchers {

  def softmax(x: DenseVector[Real])
  : DenseVector[Real] = {
    val sum: Real = Real(VectorEx.foldLeft(0.0, x)(_ + Math.exp(_)))
    DenseVector(VectorEx.map(x)(
      x => Real(Math.exp(x) / sum)
    ))
  }

  def softmax(x: DenseMatrix[Real])
  : DenseMatrix[Real] = {
    val result = DenseMatrix.zeros[Real](x.rows, x.cols)
    MatrixEx.foreachColumnVector(result, x)(_ := softmax(_))
    result
  }

  def softmax_dx(x: DenseVector[Real], err: DenseVector[Real])
  : DenseVector[Real] = {
    val y = softmax(x)
    val sum = VectorEx.dot(err, y)
    val tmp = err - sum
    tmp :*= y
    tmp
  }

  def softmax_dx(x: DenseMatrix[Real], err: DenseMatrix[Real])
  : DenseMatrix[Real] = {
    val result = DenseMatrix.zeros[Real](x.rows, x.cols)
    MatrixEx.foreachColumnVector(result, x, err)(_ := softmax_dx(_, _))
    result
  }

  "All Softmax variants" should "behave as exactly as defined" in {

    val x = RealArrayTensor.fill(Size1(2, 1), 4, PseudoRNG.default.gaussianDistribution())
    val y = RealArrayTensor.derive(Size1(2, 1), softmax(x.valuesMatrix))
    val e = RealArrayTensor.fill(Size1(2, 1), 4, PseudoRNG.default.gaussianDistribution())
    val d = RealArrayTensor.derive(Size1(2, 1), softmax_dx(x.valuesMatrix, e.valuesMatrix))

    // Isolate CPU Baseline variant.
    if (true) {
      SoftmaxBuilder.unregisterAll()
      SoftmaxBuilder.register(10, Softmax_JVM_Baseline_Description)
      val layer = SoftmaxBuilder().build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      similarity(y, prediction.output) should be < tolerance0
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      similarity(d, dHat) should be < tolerance0
    }

    // Isolate CPU ApacheCommons variant.
    if (true) {
      SoftmaxBuilder.unregisterAll()
      SoftmaxBuilder.register(10, Softmax_JVM_ApacheCommons_Description)
      val layer = SoftmaxBuilder().build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      similarity(y, prediction.output) should be < tolerance0
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      similarity(d, dHat) should be < tolerance0
    }

    // Isolate CUDA Direct variant.
    if (true) {
      SoftmaxBuilder.unregisterAll()
      SoftmaxBuilder.register(10, Softmax_CUDA_CUDNN_Description)
      val layer = SoftmaxBuilder().build(BuildHints.derive(x))
      val prediction = layer.predict(Training(0L), x, null)
      similarity(y, prediction.output) should be < tolerance0
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      similarity(d, dHat) should be < tolerance0
    }
  }

}
