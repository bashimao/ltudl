/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2014 Matthias Langer (t3l@threelights.de)
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
 */

package edu.latrobe.kernels

import edu.latrobe._
import edu.latrobe.sizes._
import scala.util.hashing._
import spire.implicits._

final class Kernel3(override val size:     (Int, Int, Int),
                    override val stride:   (Int, Int, Int),
                    override val padding0: (Int, Int, Int),
                    override val padding1: (Int, Int, Int))
  extends Kernel(size._1 * size._2 * size._3)
    with CartesianKernel[(Int, Int, Int)] {
  require(size._1 > 0)
  require(size._2 > 0)
  require(size._3 > 0)
  require(stride._1 >= 0)
  require(stride._2 >= 0)
  require(stride._3 >= 0)
  require(padding0._1 < size._1 && padding1._1 < size._1)
  require(padding0._2 < size._2 && padding1._2 < size._2)
  require(padding0._3 < size._3 && padding1._3 < size._3)

  override def toString
  : String = s"Kernel3[$size, $stride, $padding0, $padding1]"

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Kernel3]

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, size.hashCode())
    tmp = MurmurHash3.mix(tmp, stride.hashCode())
    tmp = MurmurHash3.mix(tmp, padding0.hashCode())
    tmp = MurmurHash3.mix(tmp, padding1.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Kernel3 =>
      size     == other.size     &&
      stride   == other.stride   &&
      padding0 == other.padding0 &&
      padding1 == other.padding1
    case _ =>
      false
  })

  override val ensuresAllValid: Boolean = {
    val x = padding0._1 <= 0 && padding1._1 <= 0
    val y = padding0._2 <= 0 && padding1._2 <= 0
    val z = padding0._3 <= 0 && padding1._3 <= 0
    x && y && z
  }

  override def isCentered: Boolean = {
    val x = CartesianKernel.isCentered(size._1, padding0._1, padding1._1)
    val y = CartesianKernel.isCentered(size._2, padding0._2, padding1._2)
    val z = CartesianKernel.isCentered(size._3, padding0._3, padding1._3)
    x && y && z
  }

  override def hasUnitStride
  : Boolean = stride._1 == 1 && stride._2 == 1 && stride._3 == 1

  val noValuesPerPlane: Int = size._1 * size._2


  // ---------------------------------------------------------------------------
  //     Pair index conversion related.
  // ---------------------------------------------------------------------------
  override def localPairNoOf(localPairPos: (Int, Int, Int)): Int = {
    require(localPairPos._1 >= 0 && localPairPos._1 < size._1)
    require(localPairPos._2 >= 0 && localPairPos._2 < size._2)
    require(localPairPos._3 >= 0 && localPairPos._3 < size._3)
    val x = localPairPos._1
    val y = localPairPos._2 * size._1
    val z = localPairPos._3 * noValuesPerPlane
    x + y + z
  }

  override def localPairPositionOf(localPairNo: Int): (Int, Int, Int) = {
    require(localPairNo >= 0 && localPairNo < noValues)
    val z    = localPairNo / noValuesPerPlane
    val zRem = localPairNo % noValuesPerPlane
    val y    = zRem / size._1
    val x    = zRem % size._1
    (x, y, z)
  }

  override def localPairPositionOfCenterPair
  : (Int, Int, Int) = (size._1 / 2, size._2 / 2, size._3 / 2)


  // ---------------------------------------------------------------------------
  //    Offset lookup.
  // ---------------------------------------------------------------------------
  override def offsetOfFirstPair(inputSize: Size): Int = inputSize match {
    case inputSize: Size3 => offsetOfFirstPair(inputSize)
    case inputSize: Size4 => offsetOfFirstPair(inputSize)
  }

  def offsetOfFirstPair(inputSize: Size3): Int = {
    val inpStride = inputSize.stride
    val x = padding0._1 * inpStride._1
    val y = padding0._2 * inpStride._2
    val z = padding0._3 * inpStride._3
    -x - y - z
  }

  def offsetOfFirstPair(inputSize: Size4): Int = {
    val inpStride = inputSize.stride
    val x = padding0._1 * inpStride._1
    val y = padding0._2 * inpStride._2
    val z = padding0._3 * inpStride._3
    -x - y - z
  }


  /*
  override def offsetOfFirstPairOf(inputSize: SizeLike, instanceNo: Int)
  : Int = inputSize match {
    case inputSize: Size3 => offsetOfFirstPairOf(inputSize, instanceNo)
  }

  def offsetOfFirstPairOf(inputSize: Size3, instanceNo: Int): Int = {
    debug_req(instanceNo >= 0)
    val a = offsetOfFirstPair(inputSize)
    val b = {
      val outputWidth = noOutputsForCallback(
        inputSize.width, size.width, stride._1, padding._1
      )
      val outputHeight = noOutputsForCallback(
        inputSize.height, size.height, stride._2, padding._2
      )
      val outputDepth = noOutputsForCallback(
        inputSize.depth, size.depth, stride._3, padding._3
      )
      val wh  = outputWidth * outputHeight
      debug_req(instanceNo < wh * outputDepth)
      val z   = instanceNo % wh
      val tmp = instanceNo / wh
      val y   = tmp / outputWidth
      val x   = tmp % outputWidth
      x * stride._1 + y * inputSize.width + z * inputSize.noValuesPerPlane
    }
    a + b
  }
*/
  /*
  override def offsetOf(position: (Int, Int, Int), baseOffset: Int): Int = {
    offsetOfFirst(baseOffset) +
      stride._1 * position._1 +
      stride._2 * position._2 * inputSize._1 +
      stride._3 * position._3 * inputSize._1 * inputSize._2
  }
*/

  override def relativeFirstOffsetOfPairOf(inputSize:   Size,
                                           localPairNo: Int)
  : Int = inputSize match {
    case inputSize: Size3 => relativeOffsetOfPairOf(inputSize, localPairNo)
    case inputSize: Size4 => relativeOffsetOfPairOf(inputSize, localPairNo)
  }

  def relativeOffsetOfPairOf(inputSize: Size3, localPairNo: Int)
  : Int = relativeOffsetOfPairOf(inputSize, localPairPositionOf(localPairNo))

  def relativeOffsetOfPairOf(inputSize: Size4, localPairNo: Int)
  : Int = relativeOffsetOfPairOf(inputSize, localPairPositionOf(localPairNo))

  override def relativeFirstOffsetOfPairOf(inputSize:         Size,
                                           localPairPosition: (Int, Int, Int))
  : Int = inputSize match {
    case inputSize: Size3 => relativeOffsetOfPairOf(inputSize, localPairPosition)
    case inputSize: Size4 => relativeOffsetOfPairOf(inputSize, localPairPosition)
  }

  def relativeOffsetOfPairOf(inputSize:         Size3,
                             localPairPosition: (Int, Int, Int))
  : Int = {
    require(localPairPosition._1 >= 0 && localPairPosition._1 < size._1)
    require(localPairPosition._2 >= 0 && localPairPosition._2 < size._2)
    require(localPairPosition._3 >= 0 && localPairPosition._3 < size._3)
    val inpStride = inputSize.stride
    val x = localPairPosition._1 * inpStride._1
    val y = localPairPosition._2 * inpStride._2
    val z = localPairPosition._3 * inpStride._3
    x + y + z
  }

  def relativeOffsetOfPairOf(inputSize:         Size4,
                             localPairPosition: (Int, Int, Int))
  : Int = {
    require(localPairPosition._1 >= 0 && localPairPosition._1 < size._1)
    require(localPairPosition._2 >= 0 && localPairPosition._2 < size._2)
    require(localPairPosition._3 >= 0 && localPairPosition._3 < size._3)
    val inpStride = inputSize.stride
    val x = localPairPosition._1 * inpStride._1
    val y = localPairPosition._2 * inpStride._2
    val z = localPairPosition._3 * inpStride._3
    x + y + z
  }


  // ---------------------------------------------------------------------------
  //    Derived metrics.
  // ---------------------------------------------------------------------------
  override def inputSizeFor(noChannels: Int)
  : Size3 = Size3(size, noChannels)

  def outputSizeFor(inputSize: Size3, noMaps: Int)
  : Size3 = doOutputSizeFor(inputSize, noMaps, CartesianKernel.outputSize)

  def outputSizeFor(inputSize: Size4, noMaps: Int)
  : Size4 = doOutputSizeFor(inputSize, noMaps, CartesianKernel.outputSize)

  override protected def doOutputSizeFor(inputSize: Size,
                                         noMaps:    Int,
                                         callback:  (Int, Int, Int, Int, Int) => Int)
  : Size = inputSize match {
    case inputSize: Size3 => doOutputSizeFor(inputSize, noMaps, callback)
    case inputSize: Size4 => doOutputSizeFor(inputSize, noMaps, callback)
  }

  protected def doOutputSizeFor(inputSize: Size3,
                                noMaps:    Int,
                                callback:  (Int, Int, Int, Int, Int) => Int)
  : Size3 = Size3(
    callback(inputSize.dims._1, size._1, stride._1, padding0._1, padding1._1),
    callback(inputSize.dims._2, size._2, stride._2, padding0._2, padding1._2),
    callback(inputSize.dims._3, size._3, stride._3, padding0._3, padding1._3),
    noMaps
  )

  protected def doOutputSizeFor(inputSize: Size4,
                                noMaps:    Int,
                                callback:  (Int, Int, Int, Int, Int) => Int)
  : Size4 = Size4(
    callback(inputSize.dims._1, size._1, stride._1, padding0._1, padding1._1),
    callback(inputSize.dims._2, size._2, stride._2, padding0._2, padding1._2),
    callback(inputSize.dims._3, size._3, stride._3, padding0._3, padding1._3),
    inputSize.dims._4,
    noMaps
  )


  // ---------------------------------------------------------------------------
  //    Iteration methods.
  // ---------------------------------------------------------------------------
  override def foreachOutput(inputSize: Size,
                             noMaps:    Int,
                             fn:        (Int, Int, Int) => Unit)
  : Unit = {
    if (ensuresAllValid) {
      inputSize match {
        case inputSize: Size3 => doForeachOutputSafe(inputSize, noMaps, fn)
        case inputSize: Size4 => doForeachOutputSafe(inputSize, noMaps, fn)
        case _ => throw new IllegalArgumentException
      }
    }
    else {
      inputSize match {
        case inputSize: Size3 => doForeachOutputUnsafe(inputSize, noMaps, fn)
        case inputSize: Size4 => doForeachOutputUnsafe(inputSize,noMaps,  fn)
        case _ => throw new IllegalArgumentException
      }
    }
  }

  protected def doForeachOutputSafe(inputSize: Size3,
                                    noMaps:    Int,
                                    fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val outputSize = outputSizeFor(inputSize, noMaps)
    doForeachOutputSafe(
      inputSize,
      outputSize,
      0,
      outputSize.noValues,
      offsetOfFirstPair(inputSize),
      fn
    )
  }

  protected def doForeachOutputSafe(inputSize: Size4,
                                    noMaps:    Int,
                                    fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val inputSize3  = inputSize.boxSize
    val outputSize3 = outputSizeFor(inputSize3, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize3.noValues * inputSize.dims._4
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize3.noValues
      doForeachOutputSafe(inputSize3, outputSize3, i0, i1, offset, fn)
      offset += inputSize3.noValues
      i0 = i1
    }
  }

  protected def doForeachOutputSafe(inputSize:  Size3,
                                    outputSize: Size3,
                                    baseIndex:  Int,
                                    endIndex:   Int,
                                    baseOffset: Int,
                                    fn:         (Int, Int, Int) => Unit)
  : Unit = {
    // Pre-compute frequently used values.
    val outStep = outputSize.stride
    val (inpStepX, gapY, gapZ) = {
      val inpStride = inputSize.stride
      val inpStepX = stride._1 * inpStride._1
      val inpStepY = stride._2 * inpStride._2
      val inpStepZ = stride._3 * inpStride._3
      val gapY = inpStepY - inpStepX * outputSize.dims._1
      val gapZ = inpStepZ - inpStepY * outputSize.dims._2
      (inpStepX, gapY, gapZ)
    }

    //val gapY = stride._2 * inputSize.width - stride._1 * outputSize.width
    //val gapZ = (stride._3 * inputSize.height - stride._2 * outputSize.height) * inputSize.width

    // Move kernel through input.
    var offset = baseOffset
    var i0     = baseIndex
    while (i0 < endIndex) {
      val nextGapZ = i0 + outStep._3
      while (i0 < nextGapZ) {
        val nextGapY = i0 + outStep._2
        while (i0 < nextGapY) {
          val i1 = i0 + outStep._1
          fn(i0, i1, offset)
          offset += inpStepX
          i0 = i1
        }
        offset += gapY
      }
      offset += gapZ
    }
  }

  protected def doForeachOutputUnsafe(inputSize: Size3,
                                      noMaps:    Int,
                                      fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val outputSize = outputSizeFor(inputSize, noMaps)
    doForeachOutputUnsafe(
      inputSize,
      outputSize,
      0,
      outputSize.noValues,
      offsetOfFirstPair(inputSize),
      fn
    )
  }

  protected def doForeachOutputUnsafe(inputSize: Size4,
                                      noMaps:    Int,
                                      fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val inputSize3  = inputSize.boxSize
    val outputSize3 = outputSizeFor(inputSize3, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize3.noValues * inputSize.dims._4
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize3.noValues
      doForeachOutputUnsafe(inputSize3, outputSize3, i0, i1, offset, fn)
      offset += inputSize3.noValues
      i0 = i1
    }
  }

  // TODO: Could be done faster!
  protected def doForeachOutputUnsafe(inputSize:  Size3,
                                      outputSize: Size3,
                                      baseIndex:  Int,
                                      endIndex:   Int,
                                      baseOffset: Int,
                                      fn:         (Int, Int, Int) => Unit)
  : Unit  = doForeachOutputUnsafe(
    inputSize, outputSize, baseIndex, endIndex, baseOffset,
    (i0, i1, offset0, x0, y0, z0) => fn(i0, i1, offset0)
  )

  protected def doForeachOutputUnsafe(inputSize:  Size3,
                                      outputSize: Size3,
                                      baseIndex:  Int,
                                      endIndex:   Int,
                                      baseOffset: Int,
                                      fn:         (Int, Int, Int, Int, Int, Int) => Unit)
  : Unit = {
    // Pre-compute frequently used values.
    val outStep = outputSize.stride
    val (inpStepX, gapY, gapZ) = {
      val inpStride = inputSize.stride
      val inpStepX = stride._1 * inpStride._1
      val inpStepY = stride._2 * inpStride._2
      val inpStepZ = stride._3 * inpStride._3
      val gapY = inpStepY - inpStepX * outputSize.dims._1
      val gapZ = inpStepZ - inpStepY * outputSize.dims._2
      (inpStepX, gapY, gapZ)
    }
    //val strideY = stride._2 * inputSize.width
    //val strideZ = stride._3 * inputSize.noValuesPerPlane

    // Move kernel through input.
    var z0     = -padding0._3
    var offset = baseOffset
    var i0     = baseIndex
    while (i0 < endIndex) {
      var y0       = -padding0._2
      val nextGapZ = i0 + outStep._3
      while (i0 < nextGapZ) {
        var x0       = -padding0._1
        val nextGapY = i0 + outStep._2
        while (i0 < nextGapY) {
          val i1 = i0 + outStep._1
          fn(i0, i1, offset, x0, y0, z0)
          offset += inpStepX
          x0     += stride._1
          i0 = i1
        }
        offset += gapY
        y0     += stride._2
      }
      offset += gapZ
      z0     += stride._3
    }
  }

  override def foreachValidPairEx(inputSize: Size,
                                  noMaps:    Int,
                                  fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    if (ensuresAllValid) {
      inputSize match {
        case inputSize: Size3 => doForeachValidPairExSafe(inputSize, noMaps, fn)
        case inputSize: Size4 => doForeachValidPairExSafe(inputSize, noMaps, fn)
        case _ => throw new IllegalArgumentException
      }
    }
    else {
      inputSize match {
        case inputSize: Size3 => doForeachValidPairExUnsafe(inputSize, noMaps, fn)
        case inputSize: Size4 => doForeachValidPairExUnsafe(inputSize, noMaps, fn)
        case _ => throw new IllegalArgumentException
      }
    }
  }

  protected def doForeachValidPairExSafe(inputSize: Size3,
                                         noMaps:    Int,
                                         fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    val outputSize = outputSizeFor(inputSize, noMaps)
    doForeachValidPairExSafe(
      inputSize,
      outputSize,
      0,
      outputSize.noValues,
      offsetOfFirstPair(inputSize),
      fn
    )
  }

  protected def doForeachValidPairExSafe(inputSize: Size4,
                                         noMaps:    Int,
                                         fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    val inputSize3  = inputSize.boxSize
    val outputSize3 = outputSizeFor(inputSize3, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize3.noValues * inputSize.dims._4
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize3.noValues
      doForeachValidPairExSafe(inputSize3, outputSize3, i0, i1, offset, fn)
      offset += inputSize3.noValues
      i0 = i1
    }
  }

  protected def doForeachValidPairExSafe(inputSize:  Size3,
                                         outputSize: Size3,
                                         baseIndex:  Int,
                                         endIndex:   Int,
                                         baseOffset: Int,
                                         fn:         (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    // Pre-compute frequently used values.
    val noValuesX   = size._1 * inputSize.noChannels
    val noValuesXY  = size._2 * noValuesX
    val noValuesXYZ = size._3 * noValuesXY
    val (gapY, gapZ) = {
      val inpStride = inputSize.stride
      val gapY = inpStride._2 - noValuesX//= size._1 * inpStride._1
      val gapZ = inpStride._3 - size._2 * inpStride._2
      (gapY, gapZ)
    }

    // Foreach outputs, foreach pair.
    doForeachOutputSafe(inputSize, outputSize, baseIndex, endIndex, baseOffset,
      (i0: Int, i1: Int, baseOffset: Int) => {
        val (fnPair, fnPost) = fn(i0, i1, baseOffset)

        // Cycle through kernel dimensions.
        var offset0 = baseOffset
        var j0      = 0
        while (j0 < noValuesXYZ) {
          val nextGapZ = j0 + noValuesXY
          while (j0 < nextGapZ) {
            val nextGapY = j0 + noValuesX
            while (j0 < nextGapY) {
              val j1      = j0      + inputSize.noChannels
              val offset1 = offset0 + inputSize.noChannels
              fnPair(j0, j1, offset0, offset1)
              offset0 = offset1
              j0      = j1
            }
            offset0 += gapY
          }
          offset0 += gapZ
        }

        // Call post.
        fnPost()
      }
    )
  }

  protected def doForeachValidPairExUnsafe(inputSize: Size3,
                                           noMaps:    Int,
                                           fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    val outputSize = outputSizeFor(inputSize, noMaps)
    doForeachValidPairExUnsafe(
      inputSize,
      outputSize,
      0,
      outputSize.noValues,
      offsetOfFirstPair(inputSize),
      fn
    )
  }

  protected def doForeachValidPairExUnsafe(inputSize: Size4,
                                           noMaps:    Int,
                                           fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    val inputSize3  = inputSize.boxSize
    val outputSize3 = outputSizeFor(inputSize3, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize3.noValues * inputSize.dims._4
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize3.noValues
      doForeachValidPairExUnsafe(inputSize3, outputSize3, i0, i1, offset, fn)
      offset += inputSize3.noValues
      i0 = i1
    }
  }

  protected def doForeachValidPairExUnsafe(inputSize:  Size3,
                                           outputSize: Size3,
                                           baseIndex:  Int,
                                           endIndex:   Int,
                                           baseOffset: Int,
                                           fn:         (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    // Pre-compute frequently used values.
    val maxX = inputSize.dims._1 + Math.min(padding1._1, 0)
    val maxY = inputSize.dims._2 + Math.min(padding1._2, 0)
    val maxZ = inputSize.dims._3 + Math.min(padding1._3, 0)
    val (gapY, gapZ) = {
      val inpStride = inputSize.stride
      val gapY = inpStride._2 - size._1 * inpStride._1
      val gapZ = inpStride._3 - size._2 * inpStride._2
      (gapY, gapZ)
    }
    /*
    val inpStride   = inputSize.stride
    val limitL      = inpStride._1 * Math.max(padding._1, 0)
    val limitR      = inpStride._1 * Math.min(Math.max(size._1 - padding._1, 0), size._1)
    val noValuesX   = inpStride._1 * size._1
    val limitT      = noValuesX * Math.max(padding._2, 0)
    val limitB      = noValuesX * Math.min(Math.max(size._2 - padding._2, 0), size._2)
    val noValuesXY  = noValuesX * size._2
    val limitN      = noValuesXY * Math.max(padding._3, 0)
    val limitF      = noValuesXY * Math.min(Math.max(size._3 - padding._3, 0), size._3)
    val noValuesXYZ = noValuesXY * size._3
    */

    //val gapY = inputSize.width - size.width
    //val gapZ = (inputSize.height - size.height) * inputSize.width

    // Foreach outputs, foreach pair.
    doForeachOutputUnsafe(inputSize, outputSize, baseIndex, endIndex, baseOffset,
      (i0: Int, i1: Int, baseOffset: Int, x0: Int, y0: Int, z0: Int) => {
        val (fnPair, fnPost) = fn(i0, i1, baseOffset)

        // TODO: Could be done slightly faster! (Call safe method if safe!)
        val x1 = x0 + size._1
        val y1 = y0 + size._2
        val z1 = z0 + size._3

        /*
        // If unsafe kernel instance (left, right, top, bottom, near, far)
        val begX = if (x0 == 0) limitL else 0
        val begY = if (y0 == 0) limitT else 0
        val begZ = if (z0 == 0) limitN else 0
        val endX = if (x0 == outputSize.dims._1 - 1) limitR else noValuesX
        val endY = if (y0 == outputSize.dims._2 - 1) limitB else noValuesXY
        val endZ = if (z0 == outputSize.dims._3 - 1) limitF else noValuesXYZ
        val remY = noValuesX  - endX
        val remZ = noValuesXY - endY
        val gapY = inpStride._2 - endX // Note different gaps!
        //val gapZ = (inputSize.height - yEnd) * inputSize.width
        // TODO: Can do this slightly faster!
        val gapZ = inpStride._3 - inpStride._2 * (endY / noValuesX)
        */

        // Cycle through kernel dimensions.
        var offset0 = baseOffset
        var j0      = 0
        cfor(z0)(_ < z1, _ + 1)(z => {
          cfor(y0)(_ < y1, _ + 1)(y => {
            cfor(x0)(_ < x1, _ + 1)(x => {
              val j1      = j0      + inputSize.noChannels
              val offset1 = offset0 + inputSize.noChannels
              // TODO: Could do this slightly faster!
              if (x >= 0 && x < maxX && y >= 0 && y < maxY && z >= 0 && z < maxZ) {
                fnPair(j0, j1, offset0, offset1)
              }
              offset0 = offset1
              j0      = j1
            })
            offset0 += gapY
          })
          offset0 += gapZ
        })

        // Call post.
        fnPost()
      }
    )
  }

  /*

  override def foreach(fn: (Int, Int) => Unit, index0: Int, offset0: Int)
  : Unit = {
    // Pre-compute frequently used values.
    val gapY = stride._2 * inputSize._1 - stride._1 * outputSize._1
    val dimZ = outputSize._1 * outputSize._2
    val gapZ = (stride._3 * inputSize._2 - stride._2 * outputSize._2) * inputSize._1

    // Numerical index of kernel instance.
    var offset = offsetOfFirst(offset0)
    val iEnd   = index0 + noInstances
    var i      = index0
    while (i < iEnd) {
      val nextGapZ = i + dimZ
      while (i < nextGapZ) {
        val nextGapY = i + outputSize._1
        while (i < nextGapY) {
          fn(i, offset)
          offset += stride._1
          i      += 1
        }
        offset += gapY
      }
      offset += gapZ
    }
  }

  def foreachEx(fn:      (Int, Int, Int, Int, Int) => Unit,
                index0:  Int,
                offset0: Int)
  : Unit = {
    // Pre-compute frequently used values.
    val gapY = stride._2 * inputSize._1 - stride._1 * outputSize._1
    val dimZ = outputSize._1 * outputSize._2
    val gapZ = (stride._3 * inputSize._2 - stride._2 * outputSize._2) * inputSize._1

    // Numerical index of kernel instance.
    var offset = offsetOfFirst(offset0)
    var z      = origin._3
    val iEnd   = index0 + noInstances
    var i      = index0
    while (i < iEnd) {
      var y        = origin._2
      val nextGapZ = i + dimZ
      while (i < nextGapZ) {
        // TODO: If you have nothing else to do, remove this temporary variable.
        var x        = origin._1
        val nextGapY = i + outputSize._1
        while (i < nextGapY) {
          fn(i, offset, x, y, z)
          x      += stride._1
          offset += stride._1
          i      += 1
        }
        y      += stride._2
        offset += gapY
      }
      z      += stride._3
      offset += gapZ
    }
  }

  override def foreachValidPair(fn:      (Int, Int) => (Int, Int) => Unit,
                                index0:  Int,
                                offset0: Int)
  : Unit = {
    if (noValidInstances == noInstances) {
      foreach(
        (i, offset) => foreachPairCallbackSafe(offset, fn(i, offset)),
        index0,
        offset0
      )
    }
    else {
      foreachEx((i, offset, x0, y0, z0) =>
        foreachPairCallbackUnsafe(offset, fn(i, offset), x0, y0, z0),
        index0,
        offset0
      )
    }
  }

  override def foreachValidPair(index:   Int,
                                fn:      (Int, Int) => Unit,
                                index0:  Int,
                                offset0: Int)
  : Unit = {
    if (noValidInstances == noInstances) {
      val offset = offsetOf(index, offset0)
      foreachPairCallbackSafe(offset, fn)
    }
    else {
      val position     = positionOf(index)
      val offset       = offsetOf(position, offset0)
      val (x0, y0, z0) = originOf(position)
      foreachPairCallbackUnsafe(offset, fn, x0, y0, z0)
    }
  }

  /**
   * Traditional (fast method).
   * (used if safety requirements satisfied.)
   */
  protected def foreachPairCallbackSafe(offset0: Int, fn: (Int, Int) => Unit)
  : Unit = {
    // Pre-compute frequently used values.
    val gapY = inputSize._1 - size._1
    val dimZ = size._1 * size._2
    val gapZ = (inputSize._2 - size._2) * inputSize._1

    // Cycle through kernel dimensions.
    var offset = offset0
    var j      = 0
    while (j < noValues) {
      val nextGapZ = j + dimZ
      while (j < nextGapZ) {
        val nextGapY = j + size._1
        while (j < nextGapY) {
          fn(j, offset)
          offset += 1
          j      += 1
        }
        offset += gapY
      }
      offset += gapZ
    }
  }

  /**
   * Newer more flexible method. Used if touching boundaries or unsure about
   * conditions.
   */
  protected def foreachPairCallbackUnsafe(offset0: Int,
                                          fn:      (Int, Int) => Unit,
                                          x0:      Int,
                                          y0:      Int,
                                          z0:      Int)
  : Unit = {
    // Use simplified version if safe.
    val x1 = x0 + size._1
    val y1 = y0 + size._2
    val z1 = z0 + size._3
    if (x0 >= 0 && x1 <= inputSize._1 &&
      y0 >= 0 && y1 <= inputSize._2 &&
      z0 >= 0 && z1 <= inputSize._3) {
      foreachPairCallbackSafe(offset0, fn)
    }
    else {
      // Pre-compute frequently used values.
      val gapY = inputSize._1 - size._1
      val gapZ = (inputSize._2 - size._2) * inputSize._1

      // Cycle through kernel dimensions.
      // TODO: If have nothing better to do, optimize this code. Can avoid fruitless queries right away.
      var offset = offset0
      var j = 0
      var z = z0
      while (z < z1) {
        if (z >= 0 && z < inputSize._3) {
          var y = y0
          while (y < y1) {
            if (y >= 0 && y < inputSize._2) {
              var x = x0
              while (x < x1) {
                if (x >= 0 && x < inputSize._1) {
                  fn(j, offset)
                }
                j      += 1
                offset += 1
                x      += 1
              }
              offset += gapY
              y      += 1
            }
            else {
              j      += size._1
              offset += size._1
            }
          }
        }
        else {
          j      += size._2
          offset += size._2
        }
        offset += gapZ
        z      += 1
      }
    }
  }
  */

}
/*
final class VolumetricKernelBuilder {

  /*
  override val noInputs: Int = inputSize._1 * inputSize._2 * inputSize._3

  override val noInstances: Int = outputSize._1 * outputSize._2 * outputSize._3

  override val noValues: Int = size._1 * size._2 * size._3

  override val noValidInstances: Int = {
    // X dimension
    val x = {
      if (stride._1 > 0) {
        inferOutputSize(
          inputSize._1, size._1, stride._1, Math.min(padding._1, 0)
        )
      }
      else if (padding._1 <= 0 && size._1 - padding._1 <= inputSize._1) {
        outputSize._1
      }
      else {
        0
      }
    }
    // Y dimension
    val y = {
      if (stride._2 > 0) {
        inferOutputSize(
          inputSize._2, size._2, stride._2, Math.min(padding._2, 0)
        )
      }
      else if (padding._2 <= 0 && size._2 - padding._2 <= inputSize._2) {
        outputSize._2
      }
      else {
        0
      }
    }
    // Z dimension
    val z = {
      if (stride._3 > 0) {
        inferOutputSize(
          inputSize._3, size._3, stride._3, Math.min(padding._3, 0)
        )
      }
      else if (padding._3 <= 0 && size._3 - padding._3 <= inputSize._3) {
        outputSize._3
      }
      else {
        0
      }
    }
    x * y * z
  }

  override val origin: (Int, Int, Int) = (-padding._1, -padding._2, -padding._3)
*/



  // ---------------------------------------------------------------------------
  //     Pair index conversion related.
  // ---------------------------------------------------------------------------


  /*
  override val localPairIndexOfCenterPair: Int = {
    val tmp = localPairPositionOfCenterPair
    tmp._1 +
      tmp._2 * size._1 +
      tmp._3 * size._1 * size._2
  }

  override def localPairIndexOf(localPairPosition: (Int, Int, Int)): Int = {
    localPairPosition._1 +
      localPairPosition._2 * size._1 +
      localPairPosition._3 * size._1 * size._2
  }

  override def localPairPositionOf(localPairIndex: Int): (Int, Int, Int) = {
    val tmp = size._1 * size._2
    val z   = localPairIndex / tmp
    val xy  = localPairIndex % tmp
    val y   = xy / size._1
    val x   = xy % size._1
    (x, y, z)
  }

  override def localPairPositionOfCenterPair: (Int, Int, Int) = {
    val x =
    val y =
    val z =

  }
*/

  // ---------------------------------------------------------------------------
  //    Position conversion related.
  // ---------------------------------------------------------------------------
  /*
  override def indexOf(position: (Int, Int, Int)): Int = {
    position._1 +
      position._2 * outputSize._1 +
      position._3 * outputSize._1 * outputSize._2
  }

  override def positionOf(index: Int): (Int, Int, Int) = {
    val tmp = outputSize._1 * outputSize._2
    val z   = index / tmp
    val xy  = index % tmp
    val y   = xy    / outputSize._1
    val x   = xy    % outputSize._1
    (x, y, z)
  }

  override def originOf(index: Int)
  : (Int, Int, Int) = originOf(positionOf(index))

  override def originOf(position: (Int, Int, Int)): (Int, Int, Int) = {
    val x = origin._1 + stride._1 * position._1
    val y = origin._2 + stride._2 * position._2
    val z = origin._3 + stride._3 * position._3
    (x, y, z)
  }

  override def destinationOf(index: Int)
  : (Int, Int, Int) = destinationOf(positionOf(index))

  override def destinationOf(position: (Int, Int, Int)): (Int, Int, Int) = {
    val tmp = originOf(position)
    val x = tmp._1 + size._1
    val y = tmp._2 + size._2
    val z = tmp._3 + size._3
    (x, y, z)
  }
*/

}
*/

object Kernel3 {

  final def apply(width: Int, height: Int, depth: Int)
  : Kernel3 = apply((width, height, depth))

  final def apply(size: (Int, Int, Int))
  : Kernel3 = apply(size, (1, 1, 1))

  final def apply(size:   (Int, Int, Int),
                  stride: (Int, Int, Int))
  : Kernel3 = apply(size, stride, (0, 0, 0))

  final def apply(size:    (Int, Int, Int),
                  stride:  (Int, Int, Int),
                  padding: (Int, Int, Int))
  : Kernel3 = apply(size, stride, padding, padding)

  final def apply(size:     (Int, Int, Int),
                  stride:   (Int, Int, Int),
                  padding0: (Int, Int, Int),
                  padding1: (Int, Int, Int))
  : Kernel3 = new Kernel3(size, stride, padding0, padding1)

  final def centered(width: Int, height: Int, depth: Int)
  : Kernel3 = centered((width, height, depth))

  final def centered(size: (Int, Int, Int))
  : Kernel3 = centered(size, (1, 1, 1))

  final def centered(size:   (Int, Int, Int),
                     stride: (Int, Int, Int))
  : Kernel3 = apply(
    size,
    stride,
    (size._1 / 2, size._2 / 2, size._3 / 2),
    ((size._1 - 1) / 2, (size._2 - 1) / 2, (size._3 - 1) / 2)
  )

}
