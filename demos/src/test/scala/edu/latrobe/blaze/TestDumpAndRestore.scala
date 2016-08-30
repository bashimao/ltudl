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

import java.io.{ObjectInputStream, ObjectOutputStream}

import edu.latrobe._
import edu.latrobe.blaze.initializers.UniformDistributionBuilder
import edu.latrobe.blaze.modules.{LinearFilterBuilder, SequenceBuilder, SquaredErrorConstraintBuilder}
import edu.latrobe.cublaze.{CUDA, CUBlaze, LogicalDevice}
import edu.latrobe.io.LocalFileHandle
import it.unimi.dsi.fastutil.io._
import org.json4s.jackson.JsonMethods
import org.scalatest._

final class TestDumpAndRestore
  extends FlatSpec with Matchers {
  CUBlaze.unload()

  val rng = PseudoRNG.default

  val cudaDevice = LogicalDevice.claim()

  "Dumping and restoring parameter buffers across platforms" should "have no side effects" in {
    val inputLayout  = IndependentTensorLayout.derive(7, 5)
    val outputLayout = IndependentTensorLayout.derive(2, 5)

    val modelBuilder = SequenceBuilder(
      LinearFilterBuilder(outputLayout.size.noChannels),
      SquaredErrorConstraintBuilder()
    )

    val input  = RealArrayTensor.fill(inputLayout, rng.uniformDistribution())
    val output = RealArrayTensor.fill(outputLayout, rng.uniformDistribution())

    val text = JsonMethods.pretty(input.toJson)
    (LocalFileHandle.root ++ "tmp" ++ "x.json").writeText(text)

    val jvmHints = BuildHints(JVM, inputLayout)
    val jvmModel = modelBuilder.build(jvmHints)
    jvmModel.refresh()
    jvmModel.reset(UniformDistributionBuilder().build(InstanceSeed.default))

    val cudaHints = BuildHints(CUDA, inputLayout)
    val cudaModel = modelBuilder.build(cudaHints)
    jvmModel.reset(UniformDistributionBuilder().build(InstanceSeed.default))
    cudaModel.refresh()

    val cudaDump = {
      val w = cudaModel.weightBuffer.asOrToRealTensorBufferJVM
      using(new FastByteArrayOutputStream)(stream => {
        using(new ObjectOutputStream(stream))(stream => {
          stream.writeObject(w)
        })
        stream.trim()
        stream.array
      })
    }

    val out0Jvm  = jvmModel.predict(Inference(), input, output).dropIntermediates().output.asOrToRealArrayTensor
    val out0Cuda = cudaModel.predict(Inference(), input, output).dropIntermediates().output.asOrToRealArrayTensor

    val comp0 = out0Jvm == out0Cuda

    val cudaRestored = {
      using(new FastByteArrayInputStream(cudaDump))(stream => {
        using(new ObjectInputStream(stream))(stream => {
          stream.readObject().asInstanceOf[ValueTensorBuffer]
        })
      })
    }
    jvmModel.weightBuffer := cudaRestored

    val out1Jvm  = jvmModel.predict(Inference(), input, output).dropIntermediates().output.asOrToRealArrayTensor
    val out1Cuda = cudaModel.predict(Inference(), input, output).dropIntermediates().output.asOrToRealArrayTensor

    val comp1 = out1Jvm == out1Cuda

    val dummy = 0

    val s = jvmModel.state


    val dummy2 = 0

    //val comp = jvmDump == cudaDump
    //comp should be(true)
  }

}
