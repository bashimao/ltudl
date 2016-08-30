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

final class Kernel2(override val size:     (Int, Int),
                    override val stride:   (Int, Int),
                    override val padding0: (Int, Int),
                    override val padding1: (Int, Int))
  extends Kernel(size._1 * size._2)
    with CartesianKernel[(Int, Int)] {
  require(size._1 > 0)
  require(size._2 > 0)
  require(stride._1 >= 0)
  require(stride._2 >= 0)
  require(padding0._1 < size._1 && padding1._1 < size._1)
  require(padding0._2 < size._2 && padding1._2 < size._2)

  override def toString
  : String = s"Kernel2[$size, $stride, $padding0, $padding1]"

  override def canEqual(that: Any): Boolean = that.isInstanceOf[Kernel2]

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
    case other: Kernel2 =>
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
    x && y
  }

  override def isCentered: Boolean = {
    val x = CartesianKernel.isCentered(size._1, padding0._1, padding1._1)
    val y = CartesianKernel.isCentered(size._2, padding0._2, padding1._2)
    x && y
  }

  override def hasUnitStride: Boolean = stride._1 == 1 && stride._2 == 1


  // ---------------------------------------------------------------------------
  //    Pair number conversion related.
  // ---------------------------------------------------------------------------
  override def localPairNoOf(localPairPos: (Int, Int)): Int = {
    require(localPairPos._1 >= 0 && localPairPos._1 < size._1)
    require(localPairPos._2 >= 0 && localPairPos._2 < size._2)
    localPairPos._1 + localPairPos._2 * size._1
  }

  override def localPairPositionOf(localPairNo: Int): (Int, Int) = {
    require(localPairNo >= 0 && localPairNo < noValues)
    (localPairNo % size._1, localPairNo / size._1)
  }

  override def localPairPositionOfCenterPair
  : (Int, Int) = (size._1 / 2, size._2 / 2)


  // ---------------------------------------------------------------------------
  //    Offset lookup.
  // ---------------------------------------------------------------------------
  override def offsetOfFirstPair(inputSize: Size): Int = inputSize match {
    case inputSize: Size2 => offsetOfFirstPair(inputSize)
    case inputSize: Size3 => offsetOfFirstPair(inputSize)
    case inputSize: Size4 => offsetOfFirstPair(inputSize)
  }

  def offsetOfFirstPair(inputSize: Size2): Int = {
    val x = padding0._1 * inputSize.noChannels
    val y = padding0._2 * inputSize.strideY
    -x - y
  }

  def offsetOfFirstPair(inputSize: Size3): Int = {
    val x = padding0._1 * inputSize.noChannels
    val y = padding0._2 * inputSize.strideY
    -x - y
  }

  def offsetOfFirstPair(inputSize: Size4): Int = {
    val x = padding0._1 * inputSize.noChannels
    val y = padding0._2 * inputSize.strideY
    -x - y
  }

  /*
  override def offsetOfFirstPairOf(inputSize: SizeLike, instanceNo: Int)
  : Int = inputSize match {
    case inputSize: Size2 => offsetOfFirstPairOf(inputSize, instanceNo)
    case inputSize: Size3 => offsetOfFirstPairOf(inputSize, instanceNo)
  }

  def offsetOfFirstPairOf(inputSize: Size2, instanceNo: Int): Int = {
    debug_req(instanceNo >= 0)
    val a = offsetOfFirstPair(inputSize)
    val b = {
      val outputWidth = noOutputsForCallback(
        inputSize.width, size.width, stride._1, padding._1
      )
      val outputHeight = noOutputsForCallback(
        inputSize.height, size.height, stride._2, padding._2
      )
      debug_req(instanceNo < outputWidth * outputHeight)
      val y = instanceNo / outputWidth
      val x = instanceNo % outputWidth
      x * stride._1 + y * inputSize.width
    }
    a + b
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
      val wh  = outputWidth * outputHeight
      debug_req(instanceNo < wh * inputSize.depth)
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
  override def offsetOf(position: (Int, Int), baseOffset: Int): Int = {
    offsetOfFirst(baseOffset) +
      stride._1 * position._1 +
      stride._2 * position._2 * inputSize._1
  }
*/

  override def relativeFirstOffsetOfPairOf(inputSize:   Size,
                                           localPairNo: Int)
  : Int = inputSize match {
    case inputSize: Size2 => relativeOffsetOfPairOf(inputSize, localPairNo)
    case inputSize: Size3 => relativeOffsetOfPairOf(inputSize, localPairNo)
    case inputSize: Size4 => relativeOffsetOfPairOf(inputSize, localPairNo)
  }

  def relativeOffsetOfPairOf(inputSize: Size2, localPairNo: Int)
  : Int = relativeOffsetOfPairOf(inputSize, localPairPositionOf(localPairNo))

  def relativeOffsetOfPairOf(inputSize: Size3, localPairNo: Int)
  : Int = relativeOffsetOfPairOf(inputSize, localPairPositionOf(localPairNo))

  def relativeOffsetOfPairOf(inputSize: Size4, localPairNo: Int)
  : Int = relativeOffsetOfPairOf(inputSize, localPairPositionOf(localPairNo))

  override def relativeFirstOffsetOfPairOf(inputSize:         Size,
                                           localPairPosition: (Int, Int))
  : Int = inputSize match {
    case inputSize: Size2 => relativeOffsetOfPairOf(inputSize, localPairPosition)
    case inputSize: Size3 => relativeOffsetOfPairOf(inputSize, localPairPosition)
    case inputSize: Size4 => relativeOffsetOfPairOf(inputSize, localPairPosition)
  }

  def relativeOffsetOfPairOf(inputSize: Size2, localPairPosition: (Int, Int))
  : Int = {
    require(localPairPosition._1 >= 0 && localPairPosition._1 < size._1)
    require(localPairPosition._2 >= 0 && localPairPosition._2 < size._2)
    val x = localPairPosition._1 * inputSize.noChannels
    val y = localPairPosition._2 * inputSize.strideY
    x + y
  }

  def relativeOffsetOfPairOf(inputSize: Size3, localPairPosition: (Int, Int))
  : Int = {
    require(localPairPosition._1 >= 0 && localPairPosition._1 < size._1)
    require(localPairPosition._2 >= 0 && localPairPosition._2 < size._2)
    val x = localPairPosition._1 * inputSize.noChannels
    val y = localPairPosition._2 * inputSize.strideY
    x + y
  }

  def relativeOffsetOfPairOf(inputSize: Size4, localPairPosition: (Int, Int))
  : Int = {
    require(localPairPosition._1 >= 0 && localPairPosition._1 < size._1)
    require(localPairPosition._2 >= 0 && localPairPosition._2 < size._2)
    val x = localPairPosition._1 * inputSize.noChannels
    val y = localPairPosition._2 * inputSize.strideY
    x + y
  }


  // ---------------------------------------------------------------------------
  //    Derived metrics.
  // ---------------------------------------------------------------------------
  override def inputSizeFor(noChannels: Int)
  : Size2 = Size2(size, noChannels)

  def outputSizeFor(inputSize: Size2, noMaps: Int)
  : Size2 = doOutputSizeFor(inputSize, noMaps, CartesianKernel.outputSize)

  def outputSizeFor(inputSize: Size3, noMaps: Int)
  : Size3 = doOutputSizeFor(inputSize, noMaps, CartesianKernel.outputSize)

  def outputSizeFor(inputSize: Size4, noMaps: Int)
  : Size4 = doOutputSizeFor(inputSize, noMaps, CartesianKernel.outputSize)

  override protected def doOutputSizeFor(inputSize: Size,
                                         noMaps:    Int,
                                         callback:  (Int, Int, Int, Int, Int) => Int)
  : Size = inputSize match {
    case inputSize: Size2 => doOutputSizeFor(inputSize, noMaps, callback)
    case inputSize: Size3 => doOutputSizeFor(inputSize, noMaps, callback)
    case inputSize: Size4 => doOutputSizeFor(inputSize, noMaps, callback)
  }

  protected def doOutputSizeFor(inputSize: Size2,
                                noMaps:    Int,
                                callback:  (Int, Int, Int, Int, Int) => Int)
  : Size2 = Size2(
    callback(inputSize.dims._1, size._1, stride._1, padding0._1, padding1._1),
    callback(inputSize.dims._2, size._2, stride._2, padding0._2, padding1._2),
    noMaps
  )

  protected def doOutputSizeFor(inputSize: Size3,
                                noMaps:    Int,
                                callback:  (Int, Int, Int, Int, Int) => Int)
  : Size3 = Size3(
    callback(inputSize.dims._1, size._1, stride._1, padding0._1, padding1._1),
    callback(inputSize.dims._2, size._2, stride._2, padding0._2, padding1._2),
    inputSize.dims._3,
    noMaps
  )

  protected def doOutputSizeFor(inputSize: Size4,
                                noMaps:    Int,
                                callback:  (Int, Int, Int, Int, Int) => Int)
  : Size4 = Size4(
    callback(inputSize.dims._1, size._1, stride._1, padding0._1, padding1._1),
    callback(inputSize.dims._2, size._2, stride._2, padding0._2, padding1._2),
    inputSize.dims._3,
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
        case inputSize: Size2 => doForeachOutputSafe(inputSize, noMaps, fn)
        case inputSize: Size3 => doForeachOutputSafe(inputSize, noMaps, fn)
        case inputSize: Size4 => doForeachOutputSafe(inputSize, noMaps, fn)
        case _ => throw new IllegalArgumentException
      }
    }
    else {
      inputSize match {
        case inputSize: Size2 => doForeachOutputUnsafe(inputSize, noMaps, fn)
        case inputSize: Size3 => doForeachOutputUnsafe(inputSize, noMaps, fn)
        case inputSize: Size4 => doForeachOutputUnsafe(inputSize, noMaps, fn)
        case _ => throw new IllegalArgumentException
      }
    }
  }

  protected def doForeachOutputSafe(inputSize: Size2,
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

  protected def doForeachOutputSafe(inputSize: Size3,
                                    noMaps:    Int,
                                    fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val inputSize2  = inputSize.planeSize
    val outputSize2 = outputSizeFor(inputSize2, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize2.noValues * inputSize.dims._3
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize2.noValues
      doForeachOutputSafe(inputSize2, outputSize2, i0, i1, offset, fn)
      offset += inputSize2.noValues
      i0 = i1
    }
  }

  protected def doForeachOutputSafe(inputSize: Size4,
                                    noMaps:    Int,
                                    fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val inputSize2  = inputSize.planeSize
    val outputSize2 = outputSizeFor(inputSize2, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize2.noValues * inputSize.dims._3 * inputSize.dims._4
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize2.noValues
      doForeachOutputSafe(inputSize2, outputSize2, i0, i1, offset, fn)
      offset += inputSize2.noValues
      i0 = i1
    }
  }

  protected def doForeachOutputSafe(inputSize:  Size2,
                                    outputSize: Size2,
                                    baseIndex:  Int,
                                    endIndex:   Int,
                                    baseOffset: Int,
                                    fn:         (Int, Int, Int) => Unit)
  : Unit = {
    // Pre-compute frequently used values.
    val outStepX = outputSize.noChannels
    val outStepY = outputSize.strideY
    val inpStepX = stride._1 * inputSize.noChannels
    val gapY     = {
      val inpStepY = stride._2 * inputSize.strideY
      inpStepY - inpStepX * outputSize.dims._1
    }

    // Move kernel through input.
    var offset = baseOffset
    var i0     = baseIndex
    while (i0 < endIndex) {
      val nextGapY = i0 + outStepY
      while (i0 < nextGapY) {
        val i1 = i0 + outStepX
        fn(i0, i1, offset)
        offset += inpStepX
        i0 = i1
      }
      offset += gapY
    }
  }

  protected def doForeachOutputUnsafe(inputSize: Size2,
                                      noMaps:    Int,
                                      fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val outputSize = outputSizeFor(inputSize, noMaps)
    doForeachOutputUnsafe(
      inputSize, outputSize, 0, outputSize.noValues, offsetOfFirstPair(inputSize), fn
    )
  }

  protected def doForeachOutputUnsafe(inputSize: Size3,
                                      noMaps:    Int,
                                      fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val inputSize2  = inputSize.planeSize
    val outputSize2 = outputSizeFor(inputSize2, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize2.noValues * inputSize.dims._3
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize2.noValues
      doForeachOutputUnsafe(inputSize2, outputSize2, i0, i1, offset, fn)
      offset += inputSize2.noValues
      i0 = i1
    }
  }

  protected def doForeachOutputUnsafe(inputSize: Size4,
                                      noMaps:    Int,
                                      fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val inputSize2  = inputSize.planeSize
    val outputSize2 = outputSizeFor(inputSize2, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize2.noValues * inputSize.dims._3 * inputSize.dims._4
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize2.noValues
      doForeachOutputUnsafe(inputSize2, outputSize2, i0, i1, offset, fn)
      offset += inputSize2.noValues
      i0 = i1
    }
  }

  // TODO: Could be done faster!
  protected def doForeachOutputUnsafe(inputSize:  Size2,
                                      outputSize: Size2,
                                      baseIndex:  Int,
                                      endIndex:   Int,
                                      baseOffset: Int,
                                      fn:         (Int, Int, Int) => Unit)
  : Unit = doForeachOutputUnsafe(
    inputSize, outputSize, baseIndex, endIndex, baseOffset,
    (i0, i1, offset0, x0, y0) => fn(i0, i1, offset0)
  )

  protected def doForeachOutputUnsafe(inputSize:  Size2,
                                      outputSize: Size2,
                                      baseIndex:  Int,
                                      endIndex:   Int,
                                      baseOffset: Int,
                                      fn:         (Int, Int, Int, Int, Int) => Unit)
  : Unit = {
    // Pre-compute frequently used values.
    val outStepX = outputSize.noChannels
    val outStepY = outputSize.strideY
    val inpStepX = stride._1 * inputSize.noChannels
    val gapY     = {
      val inpStepY = stride._2 * inputSize.strideY
      inpStepY - inpStepX * outputSize.dims._1
    }

    // Move kernel through input.
    var y0     = -padding0._2
    var offset = baseOffset
    var i0     = baseIndex
    while (i0 < endIndex) {
      var x0       = -padding0._1
      val nextGapY = i0 + outStepY
      while (i0 < nextGapY) {
        val i1 = i0 + outStepX
        fn(i0, i1, offset, x0, y0)
        offset += inpStepX
        x0     += stride._1
        i0 = i1
      }
      offset += gapY
      y0     += stride._2
    }
  }

  override def foreachValidPairEx(inputSize: Size,
                                  noMaps:    Int,
                                  fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    if (ensuresAllValid) {
      inputSize match {
        case inputSize: Size2 => doForeachValidPairExSafe(inputSize, noMaps, fn)
        case inputSize: Size3 => doForeachValidPairExSafe(inputSize, noMaps, fn)
        case inputSize: Size4 => doForeachValidPairExSafe(inputSize, noMaps, fn)
        case _ => throw new IllegalArgumentException
      }
    }
    else {
      inputSize match {
        case inputSize: Size2 => doForeachValidPairExUnsafe(inputSize, noMaps, fn)
        case inputSize: Size3 => doForeachValidPairExUnsafe(inputSize, noMaps, fn)
        case inputSize: Size4 => doForeachValidPairExUnsafe(inputSize, noMaps, fn)
        case _ => throw new IllegalArgumentException
      }
    }
  }

  protected def doForeachValidPairExSafe(inputSize: Size2,
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

  protected def doForeachValidPairExSafe(inputSize: Size3,
                                         noMaps:    Int,
                                         fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    val inputSize2  = inputSize.planeSize
    val outputSize2 = outputSizeFor(inputSize2, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize2.noValues * inputSize.dims._3
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize2.noValues
      doForeachValidPairExSafe(inputSize2, outputSize2, i0, i1, offset, fn)
      offset += inputSize2.noValues
      i0 = i1
    }
  }

  protected def doForeachValidPairExSafe(inputSize: Size4,
                                         noMaps:    Int,
                                         fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    val inputSize2  = inputSize.planeSize
    val outputSize2 = outputSizeFor(inputSize2, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize2.noValues * inputSize.dims._3 * inputSize.dims._4
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize2.noValues
      doForeachValidPairExSafe(inputSize2, outputSize2, i0, i1, offset, fn)
      offset += inputSize2.noValues
      i0 = i1
    }
  }

  protected def doForeachValidPairExSafe(inputSize:  Size2,
                                         outputSize: Size2,
                                         baseIndex:  Int,
                                         endIndex:   Int,
                                         baseOffset: Int,
                                         fn:         (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    // Pre-compute frequently used values.
    val noValuesX  = size._1 * inputSize.noChannels
    val noValuesXY = size._2 * noValuesX
    val gapY       = inputSize.strideY - noValuesX // = size._1 * inpStride._1

    // Foreach outputs, foreach pair.
    doForeachOutputSafe(inputSize, outputSize, baseIndex, endIndex, baseOffset,
      (i0: Int, i1: Int, baseOffset: Int) => {
        val (fnPair, fnPost) = fn(i0, i1, baseOffset)

        // Cycle through kernel dimensions.
        var offset0 = baseOffset
        var j0      = 0
        while (j0 < noValuesXY) {
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

        // Call post.
        fnPost()
      }
    )
  }

  protected def doForeachValidPairExUnsafe(inputSize: Size2,
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

  protected def doForeachValidPairExUnsafe(inputSize: Size3,
                                           noMaps:    Int,
                                           fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    val inputSize2  = inputSize.planeSize
    val outputSize2 = outputSizeFor(inputSize2, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize2.noValues * inputSize.dims._3
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize2.noValues
      doForeachValidPairExUnsafe(inputSize2, outputSize2, i0, i1, offset, fn)
      offset += inputSize2.noValues
      i0 = i1
    }
  }

  protected def doForeachValidPairExUnsafe(inputSize: Size4,
                                           noMaps:    Int,
                                           fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    val inputSize2  = inputSize.planeSize
    val outputSize2 = outputSizeFor(inputSize2, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize2.noValues * inputSize.dims._3 * inputSize.dims._4
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize2.noValues
      doForeachValidPairExUnsafe(inputSize2, outputSize2, i0, i1, offset, fn)
      offset += inputSize2.noValues
      i0 = i1
    }
  }

  protected def doForeachValidPairExUnsafe(inputSize:  Size2,
                                           outputSize: Size2,
                                           baseIndex:  Int,
                                           endIndex:   Int,
                                           baseOffset: Int,
                                           fn:         (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    // Pre-compute frequently used values.
    val maxX = inputSize.dims._1 + Math.min(padding1._1, 0)
    val maxY = inputSize.dims._2 + Math.min(padding1._2, 0)
    val gapY = inputSize.strideY - size._1 * inputSize.noChannels
    /*
    val minX = Math.max(padding._1, 0)
    val maxX = Math.max(size._1 - minX, 0)
    val minY = Math.max(padding._2, 0)
    val maxY = Math.max(size._2 - minY, 0)
    val gapY = inputSize.strideY - size._1 * inputSize.noChannels
    */
    /*
    val noValuesX  = size._1 * inputSize.noChannels
    val noValuesXY = size._2 * noValuesX
    */

    // Foreach outputs, foreach pair.
    doForeachOutputUnsafe(inputSize, outputSize, baseIndex, endIndex, baseOffset,
      (i0: Int, i1: Int, baseOffset: Int, x0: Int, y0: Int) => {
        val (fnPair, fnPost) = fn(i0, i1, baseOffset)

        val x1 = x0 + size._1
        val y1 = y0 + size._2

        // TODO: Could be done slightly faster! (Call safe method if safe!)
        // REMARK: Wrong assumption. JVM is better at optimizing the below loop than one might think.
        /*if (x0 >= 0 && y0 >= 0 && x1 <= maxX && y1 <= maxY) {
          // Cycle through kernel dimensions.
          var offset0 = baseOffset
          var j0      = 0
          while (j0 < noValuesXY) {
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
        }
        else {*/
          // Cycle through kernel dimensions.
          var offset0 = baseOffset
          var j0      = 0
          cfor(y0)(_ < y1, _ + 1)(y => {
            cfor(x0)(_ < x1, _ + 1)(x => {
              val j1      = j0      + inputSize.noChannels
              val offset1 = offset0 + inputSize.noChannels
              // TODO: Could do this slightly faster!
              if (x >= 0 && x < maxX && y >= 0 && y < maxY) {
                fnPair(j0, j1, offset0, offset1)
              }
              offset0 = offset1
              j0      = j1
            })
            offset0 += gapY
          })
        /*}*/

        // Call post.
        fnPost()
      }
    )
  }

  /*
  def foreachInstanceSafe(fn: (Int, Int) => Unit, inputSize: VolumetricSize)
  : VolumetricSize = {
    val outputSize = outputSizeFor(inputSize)
    // Pre-compute frequently used values.
    val planeSize = outputSize.width * outputSize.height
    val yGap = stride._2 * inputSize.width - stride._1 * outputSize.width
    val zGap = inputSize.width * (inputSize.height - stride._2 * outputSize.height)

    // Move kernel through input.
    var offset = offsetOfFirstPair(inputSize)
    var i      = 0
    while (i < outputSize.noValues) {
      val nextGapZ = i + planeSize
      while (i < nextGapZ) {
        val nextGapY = i + outputSize.width
        while (i < nextGapY) {
          fn(i, offset)
          offset += stride._1
          i      += 1
        }
        offset += yGap
      }
      offset += zGap
    }
    outputSize
  }

  def foreachEx(fn: (Int, Int, Int, Int) => Unit, index0: Int, offset0: Int)
  : Unit = {
    // Pre-compute frequently used values.
    val gap = stride._2 * inputSize._1 - stride._1 * outputSize._1

    // Move kernel through input.
    var offset = offsetOfFirst(offset0)
    var y      = origin._2
    val iEnd   = index0 + noInstances
    var i      = index0
    while (i < iEnd) {
      // TODO: If you have nothing else to do, remove this temporary variable.
      // TODO: Have to add support border element replication.
      var x = origin._1
      val nextGap = i + outputSize._1
      while (i < nextGap) {
        fn(i, offset, x, y)
        x      += stride._1
        offset += stride._1
        i      += 1
      }
      y      += stride._2
      offset += gap
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
      foreachEx(
        (i, offset, x0, y0) => foreachPairCallbackUnsafe(
          offset, fn(i, offset), x0, y0
        ),
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
      val position = positionOf(index)
      val offset   = offsetOf(position, offset0)
      val (x0, y0) = originOf(position)
      foreachPairCallbackUnsafe(offset, fn, x0, y0)
    }
  }

  /**
   * Traditional (fast method).
   * (used if safety requirements satisfied.)
   */
  protected def foreachPairCallbackSafe(offset0: Int, fn: (Int, Int) => Unit)
  : Unit = {
    // Pre-compute frequently used values.
    val gap = inputSize._1 - size._1

    // Cycle through kernel dimensions.
    var offset = offset0
    var j      = 0
    while (j < noValues) {
      val nextGap = j + size._1
      while (j < nextGap) {
        fn(j, offset)
        offset += 1
        j      += 1
      }
      offset += gap
    }
  }

  /**
   * Newer more flexible method. Used if touching boundaries or unsure about
   * conditions.
   */
  protected def foreachPairCallbackUnsafe(offset0: Int,
                                          fn:      (Int, Int) => Unit,
                                          x0:      Int,
                                          y0:      Int)
  : Unit = {
    // Use simplified version if safe.
    val x1 = x0 + size._1
    val y1 = y0 + size._2
    if (x0 >= 0 && x1 <= inputSize._1 &&
      y0 >= 0 && y1 <= inputSize._2) {
      foreachPairCallbackSafe(offset0, fn)
    }
    else {
      // Pre-compute frequently used values.
      val gap  = inputSize._1 - size._1

      // Cycle through kernel dimensions.
      // TODO: If have nothing better to do, optimize this code. Can avoid fruitless queries right away.
      var offset = offset0
      var j      = 0
      var y      = y0
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
        }
        else {
          j      += size._1
          offset += size._1
        }
        offset += gap
        y      += 1
      }
    }
  }
  */

}

/*
// TODO: Add support for non-continuous kernels for multiplexed tiled resources. (call it localDelta?!)
final class SpatialKernelBuilder
  extends CartesianKernelBuilder[(Int, Int)]
    with EquatableEx[SpatialKernelBuilder] {


  /*
  override val noInstances: Int = outputSize._1 * outputSize._2

  override val noValues: Int = size._1 * size._2

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
    x * y
  }
  */

  //override val origin: (Int, Int) = (-padding._1, -padding._2)


  // ---------------------------------------------------------------------------
  //    Pair number conversion related.
  // ---------------------------------------------------------------------------



  // ---------------------------------------------------------------------------
  //    Position conversion related.
  // ---------------------------------------------------------------------------
  /*
  override def instanceNoOf(position: (Int, Int), inputSize: SizeLike): Int = {
    val outputSize = outputSizeFor(inputSize)
    outputSize.indexOf(position)
    position._1 +
      position._2 * outputSize._1
  }

  override def positionOf(index: Int): (Int, Int) = {
    val y = index / outputSize._1
    val x = index % outputSize._1
    (x, y)
  }

  override def originOf(index: Int): (Int, Int) = originOf(positionOf(index))

  override def originOf(position: (Int, Int)): (Int, Int) = {
    val x = origin._1 + stride._1 * position._1
    val y = origin._2 + stride._2 * position._2
    (x, y)
  }
  */

  /*
  override def destinationOf(index: Int)
  : (Int, Int) = destinationOf(positionOf(index))

  override def destinationOf(position: (Int, Int)): (Int, Int) = {
    val tmp = originOf(position)
    val x = tmp._1 + size._1
    val y = tmp._2 + size._2
    (x, y)
  }
  */

}
*/

object Kernel2 {

  final def apply(width: Int, height: Int)
  : Kernel2 = apply((width, height))

  final def apply(size: (Int, Int))
  : Kernel2 = apply(size, (1, 1))

  final def apply(size:   (Int, Int),
                  stride: (Int, Int))
  : Kernel2 = apply(size, stride, (0, 0))

  final def apply(size:    (Int, Int),
                  stride:  (Int, Int),
                  padding: (Int, Int))
  : Kernel2 = apply(size, stride, padding, padding)

  final def apply(size:     (Int, Int),
                  stride:   (Int, Int),
                  padding0: (Int, Int),
                  padding1: (Int, Int))
  : Kernel2 = new Kernel2(
    size, stride, padding0, padding1
  )

  final def centered(width: Int, height: Int)
  : Kernel2 = centered((width, height))

  final def centered(size: (Int, Int))
  : Kernel2 = centered(size, (1, 1))

  final def centered(size:   (Int, Int),
                     stride: (Int, Int))
  : Kernel2 = apply(
    size,
    stride,
    (size._1 / 2, size._2 / 2),
    ((size._1 - 1) / 2, (size._2 - 1) / 2)
  )

}
