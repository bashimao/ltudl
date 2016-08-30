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

import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import edu.latrobe._
import edu.latrobe.blaze.modules._
import edu.latrobe.blaze.modules.jvm._
import edu.latrobe.cublaze.modules._
import edu.latrobe.sizes._
import org.scalatest._
import TestUtils._

final class TestModule_FullyConnected extends FlatSpec with Matchers {

  "All variants" should "behave exactly as defined" in {
    val x = RealArrayTensor(Size1(1, 3), 3, Array(
      1.0f, 2.0f, 3.0f, // sample 0
      4.0f, 5.0f, 6.0f, // sample 1
      8.0f, 9.0f, 0.0f  // sample 2
    ))
    val b = DenseVector(
      -10.0f, // neuron 0
      -20.0f  // neuron 1
    )
    val w = new DenseMatrix(3, 2, Array(
      0.1f, 0.2f, 0.3f, // neuron 0
      0.4f, 0.5f, 0.6f  // neuron 1
    ))
    val y = RealArrayTensor.derive(Size1(1, 2), {
      val tmp: DenseMatrix[Real] = w.t * x.valuesMatrix
      tmp(::, *) += b
      tmp
    })
    val e: Tensor = RealArrayTensor(Size1(1, 2), 3, Array(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f))
    val d: Tensor = RealArrayTensor.derive(Size1(1, 3), {
      val tmp: DenseMatrix[Real] = w * e.valuesMatrix
      tmp
    })
    val g: DenseVector[Real] = {
      val w: DenseMatrix[Real] = x.valuesMatrix * e.valuesMatrix.t
      val b: DenseVector[Real] = sum(e.valuesMatrix(*, ::))
      DenseVector(VectorEx.concat(w.toDenseVector, b))
    }

    val builder = SequenceBuilder(
      LinearFilterBuilder(2).permuteWeightReferences(_.derive(1)),
      AddBiasBuilder().permuteWeightReferences(_.derive(2))
    )

    // Isolate Baseline variant.
    if (true) {
      LinearFilterBuilder.unregisterAll()
      LinearFilterBuilder.register(10, LinearFilter_JVM_Breeze_Description)
      val layer = builder.build(BuildHints.derive(x))
      layer.weightBuffer(0, 1) := w
      layer.weightBuffer(0, 2) := b
      layer.refresh()
      val prediction = layer.predict(Training(0L), x, null)
      similarity(y, prediction.output) should be < tolerance0
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      similarity(d, dHat) should be < tolerance0
      similarity(g, DenseVector(ArrayEx.concat(gHat(0, 1).values, gHat(0, 2).values))) should be < tolerance0
    }

    // Isolate BLAS variant.
    if (true) {
      LinearFilterBuilder.unregisterAll()
      LinearFilterBuilder.register(10, LinearFilter_JVM_BLAS_Description)
      val layer = builder.build(BuildHints.derive(x))
      layer.weightBuffer(0, 1) := w
      layer.weightBuffer(0, 2) := b
      layer.refresh()
      val prediction = layer.predict(Training(0L), x, null)
      similarity(y, prediction.output) should be < tolerance0
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      similarity(d, dHat) should be < tolerance0
      similarity(g, DenseVector(ArrayEx.concat(gHat(0, 1).values, gHat(0, 2).values))) should be < tolerance0
    }

    // Isolate CUDA Direct variant.
    if (true) {
      LinearFilterBuilder.unregisterAll()
      LinearFilterBuilder.register(10, LinearFilter_CUDA_CUBLAS_Description)
      val layer = builder.build(BuildHints.derive(x))
      layer.weightBuffer(0, 1) := w
      layer.weightBuffer(0, 2) := b
      layer.refresh()
      val prediction = layer.predict(Training(0L), x, null)
      similarity(y, prediction.output) should be < tolerance0
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute()
      similarity(d, dHat) should be < tolerance0
      similarity(g, DenseVector(ArrayEx.concat(gHat(0, 1).values, gHat(0, 2).values))) should be < tolerance0
    }

  }

}
