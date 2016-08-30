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
import edu.latrobe.cublaze._
import edu.latrobe.sizes.{Size1, Size2, Size3}
import org.scalatest._

final class TestCUDATensors
  extends FlatSpec with Matchers {
  CUBlaze.unload()

  val cudaDevice = LogicalDevice.claim().device

  val rng = PseudoRNG.default

  "a.apply(index)" should "behave the same on CPU and GPU" in {
    for (i <- 1 to 10) {

      val width      = rng.nextInt(256) + 1
      val height     = rng.nextInt(256) + 1
      val noChannels = rng.nextInt(16) + 1
      val noSamples  = rng.nextInt(10) + 1

      val sampleSize = Size2(width, height, noChannels)
      val layout     = IndependentTensorLayout(sampleSize, noSamples)

      val data = ArrayEx.fill(layout.noValues)(rng.nextReal())

      val sampleNo = rng.nextInt(noSamples)

      using(
        RealArrayTensor.zeros(layout),
        CUDARealTensor(cudaDevice, layout)
      )((cpu, gpu) => {
        cpu := data
        gpu := data
        using(
          cpu(sampleNo),
          gpu(sampleNo)
        )((cpuPart, gpuPart) => {
          val a = cpuPart.asOrToRealArrayTensor
          val b = gpuPart.asOrToRealArrayTensor

          val equality = a == b
          println(s"$layout => apply($sampleNo) => $equality")
          equality should be(true)
        })
      })
    }
  }

  "a.concat(b)" should "behave the same on CPU and GPU" in {
    for (i <- 1 to 10) {

      val width      = rng.nextInt(256) + 1
      val height     = rng.nextInt(256) + 1
      val noChannels = rng.nextInt(16) + 1
      val noSamples0 = rng.nextInt(10) + 1
      val noSamples1 = rng.nextInt(10) + 1

      val sampleSize = Size2(width, height, noChannels)
      val layout0    = IndependentTensorLayout(sampleSize, noSamples0)
      val layout1    = IndependentTensorLayout(sampleSize, noSamples1)

      val data0 = ArrayEx.fill(layout0.noValues)(rng.nextReal())
      val data1 = ArrayEx.fill(layout1.noValues)(rng.nextReal())

      using(
        RealArrayTensor.zeros(layout0),
        RealArrayTensor.zeros(layout1),
        CUDARealTensor(cudaDevice, layout0),
        CUDARealTensor(cudaDevice, layout1)
      )((cpu0, cpu1, gpu0, gpu1) => {
        cpu0 := data0
        gpu0 := data0
        cpu1 := data1
        gpu1 := data1
        using(
          cpu0.concat(cpu1),
          gpu0.concat(gpu1),
          gpu0.concat(cpu1)
        )((cpuRes, gpuRes0, gpuRes1) => {
          val a = cpuRes.asOrToRealArrayTensor
          val b = gpuRes0.asOrToRealArrayTensor
          val c = gpuRes1.asOrToRealArrayTensor

          val equality0 = a == b
          val equality1 = a == c
          println(s"($layout0).concat(${layout1.noSamples}) => $equality0 | $equality1")
          equality0 should be(true)
          equality1 should be(true)
        })
      })
    }
  }

  "a.concat(Array(...))" should "behave the same on CPU and GPU" in {
    for (i <- 1 to 10) {

      val width      = rng.nextInt(256) + 1
      val height     = rng.nextInt(256) + 1
      val noChannels = rng.nextInt(16) + 1
      val noSamples0 = rng.nextInt(10) + 1

      val sampleSize = Size2(width, height, noChannels)
      val layout0    = IndependentTensorLayout(sampleSize, noSamples0)

      val noBatches  = rng.nextInt(10)
      val layouts1   = ArrayEx.fill(noBatches)({
        val noSamples1 = rng.nextInt(10) + 1
        IndependentTensorLayout(sampleSize, noSamples1)
      })

      val data0 = ArrayEx.fill(layout0.noValues)(rng.nextReal())
      val data1 = ArrayEx.map(layouts1)(
        layout1 => ArrayEx.fill(layout1.noValues)(rng.nextReal())
      )

      using(
        RealArrayTensor.zeros(layout0),
        CUDARealTensor(cudaDevice, layout0)
      )((cpu0, gpu0) => {
        cpu0 := data0
        gpu0 := data0

        val cpu1 = ArrayEx.zip(layouts1, data1)((layout1, data1) => {
          val tmp = RealArrayTensor.zeros(layout1)
          tmp := data1
          tmp
        })

        val gpu1 = ArrayEx.zip(layouts1, data1)((layout1, data1) => {
          val tmp = CUDARealTensor(cudaDevice, layout1)
          tmp := data1
          tmp
        })

        using(
          cpu0.concat(cpu1),
          gpu0.concat(gpu1),
          gpu0.concat(cpu1)
        )((cpuRes, gpuRes0, gpuRes1) => {
          val a = cpuRes.asOrToRealArrayTensor
          val b = gpuRes0.asOrToRealArrayTensor
          val c = gpuRes1.asOrToRealArrayTensor

          val equality0 = a == b
          val equality1 = a == c
          println(s"($layout0).concat(${ArrayEx.map(layouts1)(_.noSamples).toList}) => $equality0 | $equality1")
          equality0 should be(true)
          equality1 should be(true)
        })

        ArrayEx.foreach(cpu1)(_.close())
        ArrayEx.foreach(gpu1)(_.close())
      })
    }
  }

  "a.slice(i, n)" should "behave the same on CPU and GPU" in {
    for (i <- 1 to 100) {

      val width      = rng.nextInt(256) + 1
      val height     = rng.nextInt(256) + 1
      val depth      = rng.nextInt(8) + 1
      val noChannels = rng.nextInt(16) + 1
      val noSamples  = rng.nextInt(10) + 1

      val sizeType = rng.nextInt(3) + 1
      val (sampleSize: Size, sliceUnit0: Int, sliceSize: Size) = sizeType match {
        case 1 =>
          val size       = Size1(width, noChannels)
          val tuple0     = rng.nextInt(width)
          val sliceWidth = rng.nextInt(width - tuple0) + 1
          val sliceSize  = Size1(sliceWidth, noChannels)
          (size, tuple0, sliceSize)
        case 2 =>
          val size        = Size2(width, height, noChannels)
          val tuple0      = rng.nextInt(height)
          val sliceHeight = rng.nextInt(height - tuple0) + 1
          val sliceSize   = Size2(width, sliceHeight, noChannels)
          (size, tuple0, sliceSize)
        case 3 =>
          val size       = Size3(width, height, depth, noChannels)
          val tuple0     = rng.nextInt(depth)
          val sliceDepth = rng.nextInt(depth - tuple0) + 1
          val sliceSize  = Size3(width, height, sliceDepth, noChannels)
          (size, tuple0, sliceSize)
      }
      val layout0 = IndependentTensorLayout(sampleSize, noSamples)

      val data0 = ArrayEx.fill(layout0.noValues)(rng.nextReal())
      using(
        RealArrayTensor.zeros(layout0),
        CUDARealTensor(cudaDevice, layout0)
      )((cpu0, gpu0) => {
        cpu0 := data0
        gpu0 := data0
        using(
          cpu0.slice(sliceUnit0, sliceSize),
          gpu0.slice(sliceUnit0, sliceSize)
        )((cpu1, gpu1) => {
          val a = cpu1.asOrToRealArrayTensor
          val b = gpu1.asOrToRealArrayTensor

          val bSubA = b - a

          val equality = a == b
          println(s"($layout0).slice($sliceUnit0, $sliceSize) => $equality")
          equality should be(true)
        })
      })
    }
  }

  "a.sliceChannels(i, n)" should "behave the same on CPU and GPU" in {
    for (i <- 1 to 100) {

      val width      = rng.nextInt(256) + 1
      val height     = rng.nextInt(256) + 1
      val depth      = rng.nextInt(5) + 1
      val noChannels = rng.nextInt(16) + 1
      val noSamples  = rng.nextInt(10) + 1
      val sliceNoChannels = rng.nextInt(noChannels) + 1
      val sliceChannel0   = rng.nextInt(noChannels - sliceNoChannels + 1)

      val sizeType = rng.nextInt(3) + 1
      val sampleSize0 = sizeType match {
        case 1 => Size1(width, noChannels)
        case 2 => Size2(width, height, noChannels)
        case 3 => Size3(width, height, depth, noChannels)
      }
      val layout0 = IndependentTensorLayout(sampleSize0, noSamples)

      val data0 = ArrayEx.fill(layout0.noValues)(rng.nextReal())

      using(
        RealArrayTensor.zeros(layout0),
        CUDARealTensor(cudaDevice, layout0)
      )((cpu0, gpu0) => {
        cpu0 := data0
        gpu0 := data0
        using(
          cpu0.sliceChannels(sliceChannel0, sliceNoChannels),
          gpu0.sliceChannels(sliceChannel0, sliceNoChannels)
        )((cpu1, gpu1) => {
          val a = cpu1.asOrToRealArrayTensor
          val b = gpu1.asOrToRealArrayTensor

          val bSubA = b - a

          val equality = a == b
          println(s"($layout0).sliceChannels($sliceChannel0, $sliceNoChannels) => $equality")
          equality should be(true)
        })
      })
    }
  }

  "a ++ b" should "behave the same on CPU and GPU" in {
    for (i <- 1 to 100) {

      val width0     = rng.nextInt(256) + 1
      val width1     = rng.nextInt(256) + 1
      val height0    = rng.nextInt(256) + 1
      val height1    = rng.nextInt(256) + 1
      val depth0     = rng.nextInt(3) + 1
      val depth1     = rng.nextInt(3) + 1
      val noChannels = rng.nextInt(16) + 1
      val noSamples  = rng.nextInt(10) + 1

      val sizeType = rng.nextInt(3) + 1
      val sampleSize0 = sizeType match {
        case 1 => Size1(width0, noChannels)
        case 2 => Size2(width0, height0, noChannels)
        case 3 => Size3(width0, height0, depth0, noChannels)
      }
      val sampleSize1 = sizeType match {
        case 1 => Size1(width1, noChannels)
        case 2 => Size2(width0, height1, noChannels)
        case 3 => Size3(width0, height0, depth1, noChannels)
      }
      val layout0 = IndependentTensorLayout(sampleSize0, noSamples)
      val layout1 = IndependentTensorLayout(sampleSize1, noSamples)

      val data0 = ArrayEx.fill(layout0.noValues)(rng.nextReal())
      val data1 = ArrayEx.fill(layout1.noValues)(rng.nextReal())

      using(
        RealArrayTensor.zeros(layout0),
        RealArrayTensor.zeros(layout1),
        CUDARealTensor(cudaDevice, layout0),
        CUDARealTensor(cudaDevice, layout1)
      )((cpu0, cpu1, gpu0, gpu1) => {
        cpu0 := data0
        gpu0 := data0
        cpu1 := data1
        gpu1 := data1
        using(
          cpu0 ++ cpu1,
          gpu0 ++ gpu1,
          gpu0 ++ cpu1
        )((cpuRes, gpuRes0, gpuRes1) => {
          val a = cpuRes.asOrToRealArrayTensor
          val b = gpuRes0.asOrToRealArrayTensor
          val c = gpuRes1.asOrToRealArrayTensor

          val bSubA = b - a
          val cSubA = c - a

          val equality0 = a == b
          val equality1 = a == c
          println(s"($layout0) ++ ($layout1) => $equality0 | $equality1")
          equality0 should be(true)
          equality1 should be(true)
        })
      })
    }
  }

  "a :++ b" should "behave the same on CPU and GPU" in {
    for (i <- 1 to 100) {

      val width       = rng.nextInt(256) + 1
      val height      = rng.nextInt(256) + 1
      val depth       = rng.nextInt(5) + 1
      val noChannels0 = rng.nextInt(16) + 1
      val noChannels1 = rng.nextInt(16) + 1
      val noSamples   = rng.nextInt(10) + 1

      val sizeType = rng.nextInt(3) + 1
      val sampleSize0 = sizeType match {
        case 1 => Size1(width, noChannels0)
        case 2 => Size2(width, height, noChannels0)
        case 3 => Size3(width, height, depth, noChannels0)
      }
      val sampleSize1 = sizeType match {
        case 1 => Size1(width, noChannels1)
        case 2 => Size2(width, height, noChannels1)
        case 3 => Size3(width, height, depth, noChannels1)
      }
      val layout0 = IndependentTensorLayout(sampleSize0, noSamples)
      val layout1 = IndependentTensorLayout(sampleSize1, noSamples)

      val data0 = ArrayEx.fill(layout0.noValues)(rng.nextReal())
      val data1 = ArrayEx.fill(layout1.noValues)(rng.nextReal())

      using(
        RealArrayTensor.zeros(layout0),
        RealArrayTensor.zeros(layout1),
        CUDARealTensor(cudaDevice, layout0),
        CUDARealTensor(cudaDevice, layout1)
      )((cpu0, cpu1, gpu0, gpu1) => {
        cpu0 := data0
        gpu0 := data0
        cpu1 := data1
        gpu1 := data1
        using(
          cpu0 :++ cpu1,
          gpu0 :++ gpu1,
          gpu0 :++ cpu1
        )((cpuRes, gpuRes0, gpuRes1) => {
          val a = cpuRes.asOrToRealArrayTensor
          val b = gpuRes0.asOrToRealArrayTensor
          val c = gpuRes1.asOrToRealArrayTensor

          val bSubA = b - a
          val cSubA = c - a

          val equality0 = a == b
          val equality1 = a == c
          println(s"($layout0) :++ ($layout1) => $equality0 | $equality1")
          equality0 should be(true)
          equality1 should be(true)
        })
      })
    }
  }

}
