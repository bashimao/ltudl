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
import edu.latrobe.blaze.initializers._
import edu.latrobe.blaze.gradientchecks._
import edu.latrobe.blaze.modules._
import edu.latrobe.sizes._
import org.scalatest._
import TestUtils._
import edu.latrobe.blaze.modules.generic._
import edu.latrobe.cublaze.CUBlaze
import edu.latrobe.cublaze.modules._

final class TestModule_BatchNormalization
  extends FlatSpec with Matchers {
  CUBlaze.unload()

  val variants = Array(
    BatchNormalization_Generic_Baseline_Description,
    BatchNormalization_CUDA_CUDNN_Description
  )
/*
  "BatchNormalization" should "behave as defined if gamma is one" in {
    for (variant <- variants; mode <- Array(PredictMode.forTraining(0L), PredictMode.default()); i <- 1 to 10) {
      val rng = PseudoRNG.default

      BatchNormalizationBuilder.variants.clear()
      BatchNormalizationBuilder.register(100, variant)

      val inputSize = Size2(rng.nextInt(10) + 1, rng.nextInt(10) + 1, rng.nextInt(10) + 1)
      val immSize = Size1(1, inputSize.noValues)
      val outputSize = Size1(1, rng.nextInt(10) + 1)
      val noSamples = 1
      val inputHints = BuildHints(JVM, IndependentTensorLayout(inputSize, noSamples))

      println(s"$variant: $inputSize => $immSize => $outputSize")

      val x   = RealArrayTensor.rand(inputSize, noSamples, rng.uniformDistribution(RealRange.minusOneToOne))
      val ref = RealArrayTensor.rand(outputSize, noSamples, rng.uniformDistribution(RealRange.minusOneToOne))

      val mvs = {
        val tmp = MatrixEx.reshape(x.valuesMatrix, x.layout.size.noChannels)
        MatrixEx.mapRowVectors(tmp)(row => {
          val mv = MeanAndVariance()
          VectorEx.foreach(row)(mv.update)
          mv
        })
      }

      val m0 = SequenceBuilder(
        ReshapeBuilder(immSize),
        LinearFilterBuilder(outputSize.noValues),
        AddBiasBuilder()
      ).build(inputHints)
      m0.reset(
        UniformDistributionBuilder().permuteSeeds(
          x => BuilderSeed.reproducible()
        ).forReference("filter").build()
      )
      m0.refresh()
      val x0 = x.copy
      val p0 = {
        val tmp = MatrixEx.reshape(x0.valuesMatrix, x0.layout.size.noChannels)
        MatrixEx.mapRowVectorPairs(tmp)((i, col) => {
          val mv = mvs(i)
          col -= mv.mean
          val sigma = mv.populationStdDev()
          if (sigma != Real.zero) {
            col /= sigma
          }
        })
        m0.predict(PredictMode.default(), Batch(x0))
      }
      val y0 = p0.output.asOrToRealArrayTensor

      val m1 = SequenceBuilder(
        BatchNormalizationBuilder().setHandle("gamma"),
        ReshapeBuilder(immSize),
        LinearFilterBuilder(outputSize.noValues),
        AddBiasBuilder()
      ).build(inputHints)
      m1.reset(
        UniformDistributionBuilder().permuteSeeds(
          x => BuilderSeed.reproducible()
        ).forReference("filter").build()
      )
      m1.reset(
        ConstantValueBuilder.one.forReference("gamma").build()
      )
      m1.refresh()
      val x1 = x.copy
      val p1 = m1.predict(PredictMode.forTraining(0L), Batch(x1, ref))
      val y1 = p1.output.asOrToRealArrayTensor

      TestUtils.similarity(y0, y1) should be < tolerance0

      //val err0 = m0.deriveGradients(p0, RealTensor.ones(y0.layout.toIndependentTensorLayout), -1, m0.weights.allocateSiblingAndClear()).get.asOrToRealTensor
      //val err1 = m1.deriveGradients(p1, RealTensor.ones(y1.layout.toIndependentTensorLayout), -1, m1.weights.allocateSiblingAndClear()).get.asOrToRealTensor

      val gt = ProbeEntireGroupBuilder(PredictMode.forTraining(0L)).build()

      val gtRes = gt(m1, Batch(x, ref))
      if (gtRes.rating >= tolerance0) {
        println(gtRes.rating)
        gtRes.differences.foreach(kv => println(s"${kv._1._3} -> ${kv._2._1}"))
      }
      gtRes.rating should be < tolerance0
    }
  }
*/
  "All batchNormalization variants" should "should behave the same" in {
    for (i <- 1 to 10) {
      val rng = PseudoRNG.default

      val inputSize  = Size2(1, 2, 1)// Size2(rng.nextInt(10) + 1, rng.nextInt(10) + 1, rng.nextInt(10) + 1)
      val immSize    = Size1(1, inputSize.noValues)//Size1(1, inputSize.noValues)
      val outputSize = Size1(1, 1)//Size1(1, rng.nextInt(10) + 1)
      val noSamples  = 1
      val inputHints = BuildHints(JVM, IndependentTensorLayout(inputSize, noSamples))

      println(s"$inputSize => $immSize => $outputSize")

      val x   = RealArrayTensor.fill(inputSize, noSamples, rng.uniformDistribution(RealRange.minusOneToOne))
      val ref = RealArrayTensor.fill(outputSize, noSamples, rng.uniformDistribution(RealRange.minusOneToOne))

      BatchNormalizationBuilder.unregisterAll()
      BatchNormalizationBuilder.register(100, variants(0))
      val m0 = SequenceBuilder(
        BatchNormalizationBuilder().setHandle("gamma"),
        ReshapeBuilder(immSize),
        LinearFilterBuilder(outputSize.noValues),
        AddBiasBuilder()
      ).build(inputHints)
      m0.reset(
        UniformDistributionBuilder().setSeed(
          BuilderSeed.reproducible()
        ).build()
      )
      m0.reset(FixedValueInitializerBuilder(Real.one).forReference("gamma").build())
      m0.reset(FixedValueInitializerBuilder(Real.one).build())
      m0.reset(FixedValueInitializerBuilder(Real.zero).forReference("bias").build())
      m0.refresh()
      val x0 = x.copy
      val p0 = m0.predict(Training(0L), Batch(x0))
      val y0 = p0.output.asOrToRealArrayTensor

      BatchNormalizationBuilder.unregisterAll()
      BatchNormalizationBuilder.register(100, variants(1))
      val m1 = SequenceBuilder(
        BatchNormalizationBuilder().setHandle("gamma"),
        ReshapeBuilder(immSize),
        LinearFilterBuilder(outputSize.noValues),
        AddBiasBuilder()
      ).build(inputHints)
      m1.reset(
        UniformDistributionBuilder().setSeed(
          BuilderSeed.reproducible()
        ).build()
      )
      m1.reset(FixedValueInitializerBuilder(Real.one).forReference("gamma").build())
      m1.reset(FixedValueInitializerBuilder(Real.one).build())
      m1.reset(FixedValueInitializerBuilder(Real.zero).forReference("bias").build())
      m1.refresh()
      val x1 = x.copy
      val p1 = m1.predict(Training(0L), Batch(x1, ref))
      val y1 = p1.output.asOrToRealArrayTensor

      TestUtils.similarity(y0, y1) should be < tolerance0

      val s0   = m0.weightBuffer.allocateZeroedSibling()
      val s1   = m0.weightBuffer.allocateZeroedSibling()
      val err0 = m0.deriveGradients(p0, RealArrayTensor.ones(y0.layout.makeIndependent), s0).compute().asOrToRealArrayTensor
      val err1 = m1.deriveGradients(p1, RealArrayTensor.ones(y1.layout.makeIndependent), s1).compute().asOrToRealArrayTensor
      s0 := m0.weightBuffer
      s1 := m1.weightBuffer

      TestUtils.similarity(err0, err1) should be < tolerance0

      val gt = ProbeEntireBufferBuilder().build()

      val gtRes = gt(0L, m0, Batch(x, ref))
      if (gtRes.rating >= tolerance0) {
        println(gtRes.rating)
        gtRes.differences.foreach(kv => println(s"${kv._1._3} -> ${kv._2._1}"))
      }
      gtRes.rating should be < tolerance0
    }
  }

}
