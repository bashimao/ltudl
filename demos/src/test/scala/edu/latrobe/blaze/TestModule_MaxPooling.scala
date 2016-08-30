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

import breeze.linalg.{DenseVector, max}
import edu.latrobe._
import edu.latrobe.kernels._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.sizes._
import edu.latrobe.cublaze.modules._
import org.scalatest._
import TestUtils._
import edu.latrobe.blaze.modules.jvm.MaxPooling_JVM_Baseline_Description

final class TestModule_MaxPooling extends FlatSpec with Matchers {

  "All variants of MaxPooling" should "should behave exactly as specified" in {

    // x x x x x
    // x x x x x
    // x x x x x
    val rand = PseudoRNG.default.uniformDistribution(RealRange.minusOneToOne)
    val x0 = MatrixEx.fill(15, 3, rand)

    val y_size_32_stride_21_pad_00 = VectorEx.multiConcatDenseH(
      MatrixEx.mapColumnVectors(x0)(x0 => {
        val x = VectorEx.asMatrix(x0, 5, 3)
        DenseVector(
          max(x(0 to 2, 0 to 1)), max(x(2 to 4, 0 to 1)),
          max(x(0 to 2, 1 to 2)), max(x(2 to 4, 1 to 2))
        )
      })
    )

    val y_size_32_stride_21_pad_11 = VectorEx.multiConcatDenseH(
      MatrixEx.mapColumnVectors(x0)(x0 => {
        val x = VectorEx.asMatrix(x0, 5, 3)
        DenseVector(
          max(x(0 to 1, 0 to 0)), max(x(1 to 3, 0 to 0)), max(x(3 to 4, 0 to 0)),
          max(x(0 to 1, 0 to 1)), max(x(1 to 3, 0 to 1)), max(x(3 to 4, 0 to 1)),
          max(x(0 to 1, 1 to 2)), max(x(1 to 3, 1 to 2)), max(x(3 to 4, 1 to 2)),
          max(x(0 to 1, 2 to 2)), max(x(1 to 3, 2 to 2)), max(x(3 to 4, 2 to 2))
        )
      })
    )

    val x        = RealArrayTensor.derive(Size2((5, 3), 1), x0)
    val y_pad_00 = RealArrayTensor.derive(Size2((2, 2), 1), y_size_32_stride_21_pad_00)
    val y_pad_11 = RealArrayTensor.derive(Size2((3, 4), 1), y_size_32_stride_21_pad_11)

    val k_32_21_00 = Kernel2((3, 2), (2, 1), (0, 0))
    val k_32_21_11 = Kernel2((3, 2), (2, 1), (1, 1))

    val mp_pad_00 = MaxPoolingBuilder(k_32_21_00)
    val mp_pad_11 = MaxPoolingBuilder(k_32_21_11)

    // Isolate Baseline variant.
    if (true) {
      MaxPoolingBuilder.unregisterAll()
      MaxPoolingBuilder.register(10, MaxPooling_JVM_Baseline_Description)
      val layer_00 = mp_pad_00.build(BuildHints.derive(x))
      val p_00 = layer_00.predict(Training(0L), x, y_pad_00)
      similarity(p_00.output, y_pad_00) shouldEqual Real.zero
      val layer_11 = mp_pad_11.build(BuildHints.derive(x))
      val p_11 = layer_11.predict(Training(0L), x, y_pad_11)
      similarity(p_11.output, y_pad_11) shouldEqual Real.zero
    }

    // Isolate CUDA direct variant.
    if (true) {
      MaxPoolingBuilder.unregisterAll()
      MaxPoolingBuilder.register(10, MaxPooling_CUDA_CUDNN_Description)
      val layer_00 = mp_pad_00.build(BuildHints.derive(x))
      val p_00 = layer_00.predict(Training(0L), x, y_pad_00)
      similarity(p_00.output, y_pad_00) shouldEqual Real.zero
      val layer_11 = mp_pad_11.build(BuildHints.derive(x))
      val p_11 = layer_11.predict(Training(0L), x, y_pad_11)
      similarity(p_11.output, y_pad_11) shouldEqual Real.zero
    }

  }

  it should "behave the same when confronted with multi channel inputs" in {

    val s = Size2((5, 5), 3)
    val x = RealArrayTensor.fill(s, 1, PseudoRNG.default.uniformDistribution(RealRange.minusOneToOne))

    val k_pad_00 = Kernel2((5, 5), (3, 3), (0, 0))
    val k_pad_22 = Kernel2((5, 5), (3, 3), (2, 2))

    val mp_pad_00 = MaxPoolingBuilder(k_pad_00)
    val mp_pad_22 = MaxPoolingBuilder(k_pad_22)

    // Isolate Baseline variant.
    MaxPoolingBuilder.unregisterAll()
    MaxPoolingBuilder.register(10, MaxPooling_JVM_Baseline_Description)
    val cpu_baseline_00 = {
      val m    = mp_pad_00.build(BuildHints.derive(x))
      val eval = m.predict(Training(0L), x, null)
      val tmp  = RealArrayTensor.ones(eval.output.layout.size, eval.output.layout.noSamples) * Real(2.5)
      val gHat = m.weightBuffer.allocateZeroedSibling()
      val err  = m.deriveGradients(eval, tmp, gHat)
      (eval.output, err.compute())
    }
    val cpu_baseline_22 = {
      val m    = mp_pad_22.build(BuildHints.derive(x))
      val eval = m.predict(Training(0L), x, null)
      val tmp  = RealArrayTensor.ones(eval.output.layout.size, eval.output.layout.noSamples) * Real(2.5)
      val gHat = m.weightBuffer.allocateZeroedSibling()
      val err  = m.deriveGradients(eval, tmp, gHat)
      (eval.output, err.compute())
    }

    // Isolate CUDA direct variant.
    MaxPoolingBuilder.unregisterAll()
    MaxPoolingBuilder.register(10, MaxPooling_CUDA_CUDNN_Description)
    val cuda_direct_00 = {
      val m    = mp_pad_00.build(BuildHints.derive(x))
      val eval = m.predict(Training(0L),x, null)
      val tmp  = RealArrayTensor.ones(eval.output.layout.size, eval.output.layout.noSamples) * Real(2.5)
      val gHat = m.weightBuffer.allocateZeroedSibling()
      val err  = m.deriveGradients(eval, tmp, gHat)
      (eval.output.toRealArrayTensor, err.compute().toRealArrayTensor)
    }
    val cuda_direct_22 = {
      val m    = mp_pad_22.build(BuildHints.derive(x))
      val eval = m.predict(Training(0L), x, null)
      val tmp  = RealArrayTensor.ones(eval.output.layout.size, eval.output.layout.noSamples) * Real(2.5)
      val gHat = m.weightBuffer.allocateZeroedSibling()
      val err  = m.deriveGradients(eval, tmp, gHat)
      (eval.output.toRealArrayTensor, err.compute().toRealArrayTensor)
    }

    //similarity(cpu_baseline_00._1, cuda_sandbox_00._1) shouldEqual Real.zero
    //similarity(cpu_baseline_22._1, cuda_sandbox_22._1) shouldEqual Real.zero

    //similarity(cpu_baseline_00._2, cuda_sandbox_00._2) shouldEqual Real.zero
    //similarity(cpu_baseline_22._2, cuda_sandbox_22._2) shouldEqual Real.zero


    similarity(cpu_baseline_00._1, cuda_direct_00._1) shouldEqual Real.zero
    similarity(cpu_baseline_22._1, cuda_direct_22._1) shouldEqual Real.zero

    similarity(cpu_baseline_00._2, cuda_direct_00._2) shouldEqual Real.zero
    similarity(cpu_baseline_22._2, cuda_direct_22._2) shouldEqual Real.zero
  }

}
