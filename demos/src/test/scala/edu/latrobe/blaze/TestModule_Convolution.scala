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
import edu.latrobe.kernels._
import edu.latrobe.blaze.modules._
import edu.latrobe.blaze.modules.jvm._
import edu.latrobe.cublaze.modules._
import edu.latrobe.sizes._
import org.scalatest._
import TestUtils._
import edu.latrobe.cublaze.CUBlaze

final class TestModule_Convolution
  extends FlatSpec
    with Matchers {
  CUBlaze.unload()

  "All variants" should "should behave like a fully connected layer" in {
    val x = RealArrayTensor.derive(Size1(3, 1), new DenseMatrix(3, 3, Array(
      1.0f, 2.0f, 3.0f, // sample 0
      4.0f, 5.0f, 6.0f, // sample 1
      8.0f, 9.0f, 0.0f  // sample 2
    )))
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
    val d: Tensor = RealArrayTensor.derive(Size1(1, 3), w * e.valuesMatrix)
    val g: DenseVector[Real] = {
      val b: DenseVector[Real] = sum(e.valuesMatrix(*, ::))
      val w: DenseMatrix[Real] = x.valuesMatrix * e.valuesMatrix.t
      DenseVector(VectorEx.concat(w.toDenseVector, b))
    }

    val k = Kernel1(x.layout.size.noTuples, 1)
    val c = SequenceBuilder(
      ConvolutionFilterBuilder(
        k, 2
      ).permuteWeightReferences(_.derive(1)),
      AddBiasBuilder(
      ).permuteWeightReferences(_.derive(2))
    )

    val variants = Array(
      ConvolutionFilter_JVM_Breeze_MM_Description,
      //Convolution_JVM_Baseline_SparseMM_Description,
      ConvolutionFilter_JVM_BLAS_MM_Description,
      ConvolutionFilter_JVM_BLAS_ImplicitMM_Description,
      ConvolutionFilter_CUDA_CUDNN_Description
    )

    // Isolate Baseline MM variant.
    for (variant <- variants) {
      ConvolutionFilterBuilder.unregisterAll()
      ConvolutionFilterBuilder.register(10, variant)
      val layer = c.build(BuildHints.derive(x))
      layer.weightBuffer(0, 1) := w
      layer.weightBuffer(0, 2) := b
      layer.refresh()
      val prediction = layer.predict(Training(0L), x, null)
      similarity(y, prediction.output) should be < tolerance0
      val gHat = layer.weightBuffer.allocateZeroedSibling()
      val dHat = layer.deriveGradients(prediction, e, gHat).compute().toRealArrayTensor
      val gHat2 = DenseVector(ArrayEx.concat(gHat(0, 1).values, gHat(0, 2).values))
      similarity(g, gHat2) should be < tolerance0
      similarity(d, dHat) should be < tolerance0
    }
/*
    // Isolate BLAS MM variant.
    if (true) {
      ConvolutionBuilder.unregisterAll()
      ConvolutionBuilder.register(Convolution_JVM_BLAS_MM_Description)
      val layer = FullyConnectedBuilder(2).build(BuildHints.derive(Train, x))
      layer.setWeights(0, 0, b)
      layer.setWeights(0, 2, w.asVector)
      val eval = layer.evaluate(x, null)
      similarity(y, eval.prediction.output) should be < tolerance0
      val dHat = layer.deriveInputError(eval, e).get
      similarity(d, dHat) should be < tolerance0
      val gHat = layer.deriveGradients(0, eval, e)
      similarity(g, gHat.segments.head._2) should be < tolerance0
    }

        // Isolate CUDA Sandbox variant.
        if (true) {
          ConvolutionBuilder.unregisterAll()
          ConvolutionBuilder.register(FullyConnected_CUDA_Sandbox_Description)
          val layer = FullyConnectedBuilder(2).build(BuildHints.derive(Train, x))
          layer.setWeights(0, 0, b)
          layer.setWeights(0, 2, w.asVector)
          val eval = layer.evaluate(x, null)
          similarity(y, eval.prediction.output) should be < tolerance0
          val dHat = layer.deriveInputError(eval, e).get
          similarity(d, dHat) should be < tolerance0
          val gHat = layer.deriveGradients(0, eval, e)
          similarity(g, gHat.segments.head._2) should be < tolerance0
        }*/

  }

}
