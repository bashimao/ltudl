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

import breeze.linalg.{DenseMatrix, DenseVector, axpy}
import breeze.numerics.abs
import breeze.stats.distributions.Rand
import edu.latrobe._
import edu.latrobe.Implicits._
import edu.latrobe.cublaze._
import edu.latrobe.sizes.Size2
import org.scalatest._

class TestTensors
  extends FlatSpec
    with Matchers {
  CUBlaze.unload()

  val tolerance0: Real = 1e-6f

  val tolerance1: Real = 1e-5f

  val layout = IndependentTensorLayout(
    Size2(2, 2, 3), 6
  )

  "+" should "behave the same in" in {
    val a = DenseMatrix.rand(layout.size.noValues, layout.noSamples, PseudoRNG.default.uniformDistribution())
    val b = DenseMatrix.rand(layout.size.noValues, layout.noSamples, PseudoRNG.default.uniformDistribution())
    val aPlusB = a + b

    val aJVM      = RealArrayTensor.derive(layout.size, a)
    val bJVM      = RealArrayTensor.derive(layout.size, b)
    val aPlusBJVM = (aJVM + bJVM).valuesMatrix

    val jvmDiff = aPlusB - aPlusBJVM
    all(jvmDiff.data) should be (Real.zero +- tolerance0)

    val aCUDA = CUDARealTensor(LogicalDevice.claim().device, layout)
    aCUDA := a
    val bCUDA = CUDARealTensor(LogicalDevice.claim().device, layout)
    bCUDA := b
    val aPlusBCUDA = (aCUDA + bCUDA).valuesMatrix

    val cudaDiff = aPlusB - aPlusBCUDA
    all(cudaDiff.data) should be (Real.zero +- tolerance0)
  }

  "scaleAdd(, c)" should "behave the same in" in {
    val a = DenseMatrix.rand(layout.size.noValues, layout.noSamples, PseudoRNG.default.uniformDistribution())
    val b = DenseMatrix.rand(layout.size.noValues, layout.noSamples, PseudoRNG.default.uniformDistribution())
    val c = b.copy
    val f = PseudoRNG.default.nextReal()
    val aPlusB = axpy(f, a, c)

    val aJVM = RealArrayTensor.derive(layout.size, a)
    val bJVM = RealArrayTensor.derive(layout.size, b)
    bJVM.add(aJVM, f)

    val jvmDiff = bJVM.valuesMatrix - c
    all(jvmDiff.data) should be (Real.zero +- tolerance0)

    val aCUDA = CUDARealTensor(LogicalDevice.claim().device, layout)
    aCUDA := a
    val bCUDA = CUDARealTensor(LogicalDevice.claim().device, layout)
    bCUDA := b
    bCUDA.add(aCUDA, f)

    val cudaDiff = bCUDA.valuesMatrix - c
    all(cudaDiff.data) should be (Real.zero +- tolerance0)
  }

  "unary_-" should "behave the same in" in {
    val a = DenseMatrix.rand(layout.size.noValues, layout.noSamples, PseudoRNG.default.uniformDistribution())
    val b = -a

    val aJVM = RealArrayTensor.derive(layout.size, a)
    val bJVM = -aJVM

    val jvmDiff = bJVM.valuesMatrix - b
    all(jvmDiff.data) should be (Real.zero +- tolerance0)

    val aCUDA = CUDARealTensor(LogicalDevice.claim().device, layout)
    aCUDA := a
    val bCUDA = -aCUDA

    val cudaDiff = bCUDA.valuesMatrix - b
    all(cudaDiff.data) should be (Real.zero +- tolerance0)
  }

  "-" should "behave the same in" in {
    val a = DenseMatrix.rand(layout.size.noValues, layout.noSamples, PseudoRNG.default.uniformDistribution())
    val b = DenseMatrix.rand(layout.size.noValues, layout.noSamples, PseudoRNG.default.uniformDistribution())
    val ab = a - b

    val aJVM  = RealArrayTensor.derive(layout.size, a)
    val bJVM  = RealArrayTensor.derive(layout.size, b)
    val abJVM = (aJVM - bJVM).valuesMatrix

    val jvmDiff = ab - abJVM
    all(jvmDiff.data) should be (Real.zero +- tolerance0)

    val aCUDA = CUDARealTensor(LogicalDevice.claim().device, layout)
    aCUDA := a
    val bCUDA = CUDARealTensor(LogicalDevice.claim().device, layout)
    bCUDA := b
    val abCUDA = (aCUDA - bCUDA).valuesMatrix

    val cudaDiff = ab - abCUDA
    all(cudaDiff.data) should be (Real.zero +- tolerance0)
  }

  "*" should "behave the same in" in {
    val a = DenseMatrix.rand(layout.size.noValues, layout.noSamples, PseudoRNG.default.uniformDistribution())
    val b = a * -Real.one

    val aJVM = RealArrayTensor.derive(layout.size, a)
    val bJVM = aJVM * -Real.one

    val jvmDiff = bJVM.valuesMatrix - b
    all(jvmDiff.data) should be (Real.zero +- tolerance0)

    val aCUDA = CUDARealTensor(LogicalDevice.claim().device, layout)
    aCUDA := a
    val bCUDA = aCUDA * -Real.one

    val cudaDiff = bCUDA.valuesMatrix - b
    all(cudaDiff.data) should be (Real.zero +- tolerance0)
  }

  ":*" should "behave the same in" in {
    val a = DenseMatrix.rand(layout.size.noValues, layout.noSamples, PseudoRNG.default.uniformDistribution())
    val b = DenseMatrix.rand(layout.size.noValues, layout.noSamples, PseudoRNG.default.uniformDistribution())
    val ab = a :* b

    val aJVM  = RealArrayTensor.derive(layout.size, a)
    val bJVM  = RealArrayTensor.derive(layout.size, b)
    val abJVM = (aJVM :* bJVM).valuesMatrix

    val jvmDiff = ab - abJVM
    all(jvmDiff.data) should be (Real.zero +- tolerance0)

    val aCUDA = CUDARealTensor(LogicalDevice.claim().device, layout)
    aCUDA := a
    val bCUDA = CUDARealTensor(LogicalDevice.claim().device, layout)
    bCUDA := b
    val abCUDA = (aCUDA :* bCUDA).valuesMatrix

    val cudaDiff = ab - abCUDA
    all(cudaDiff.data) should be (Real.zero +- tolerance0)
  }

  ":/" should "behave the same in" in {
    val a = DenseMatrix.rand(layout.size.noValues, layout.noSamples, PseudoRNG.default.uniformDistribution())
    val b = DenseMatrix.rand(layout.size.noValues, layout.noSamples, PseudoRNG.default.uniformDistribution())
    val ab = a :/ b

    val aJVM  = RealArrayTensor.derive(layout.size, a)
    val bJVM  = RealArrayTensor.derive(layout.size, b)
    val abJVM = (aJVM :/ bJVM).valuesMatrix

    val jvmDiff = ab - abJVM
    all(jvmDiff.data) should be (Real.zero +- tolerance0)

    val aCUDA = CUDARealTensor(LogicalDevice.claim().device, layout)
    aCUDA := a
    val bCUDA = CUDARealTensor(LogicalDevice.claim().device, layout)
    bCUDA := b
    val abCUDA = (aCUDA :/ bCUDA).valuesMatrix

    val cudaDiff = ab - abCUDA
    all(cudaDiff.data) should be (Real.zero +- tolerance0)
  }

  "dot" should "behave the same in" in {
    val a = DenseMatrix.rand(layout.size.noValues, layout.noSamples, PseudoRNG.default.uniformDistribution())
    val b = DenseMatrix.rand(layout.size.noValues, layout.noSamples, PseudoRNG.default.uniformDistribution())
    val aDotB = MatrixEx.dot(a, b)

    val aJVM      = RealArrayTensor.derive(layout.size, a)
    val bJVM      = RealArrayTensor.derive(layout.size, b)
    val aDotBJVM = aJVM.dot(bJVM)

    aDotBJVM should be (aDotB +- tolerance1)

    val aCUDA = CUDARealTensor(LogicalDevice.claim().device, layout)
    aCUDA := a
    val bCUDA = CUDARealTensor(LogicalDevice.claim().device, layout)
    bCUDA := b
    val aDotBCUDA = aCUDA.dot(bCUDA)

    aDotBCUDA should be (aDotB +- tolerance1)
  }

}
