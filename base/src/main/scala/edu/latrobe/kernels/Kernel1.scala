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

final class Kernel1(override val size:     Tuple1[Int],
                    override val stride:   Tuple1[Int],
                    override val padding0: Tuple1[Int],
                    override val padding1: Tuple1[Int])
  extends Kernel(size._1)
    with CartesianKernel[Tuple1[Int]] {
  require(size._1 > 0)
  require(stride._1 >= 0)
  require(padding0._1 < size._1)
  require(padding1._1 < size._1)

  override def toString
  : String = s"Kernel1[$size, $stride, $padding0, $padding1]"

  override def canEqual(that: Any): Boolean = that.isInstanceOf[Kernel1]

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
    case other: Kernel1 =>
      size     == other.size     &&
      stride   == other.stride   &&
      padding0 == other.padding0 &&
      padding1 == other.padding1
    case _ =>
      false
  })

  override val ensuresAllValid
  : Boolean = padding0._1 <= 0 && padding1._1 <= 0

  override def isCentered
  : Boolean = CartesianKernel.isCentered(size._1, padding0._1, padding1._1)

  override def hasUnitStride: Boolean = stride._1 == 1



  // ---------------------------------------------------------------------------
  //    Pair number conversion related.
  // ---------------------------------------------------------------------------
  override def localPairNoOf(localPairPos: Tuple1[Int]): Int = {
    require(localPairPos._1 >= 0 && localPairPos._1 < size._1)
    localPairPos._1
  }

  override def localPairPositionOf(localPairNo: Int): Tuple1[Int] = {
    require(localPairNo >= 0 && localPairNo < size._1)
    Tuple1(localPairNo)
  }

  override def localPairPositionOfCenterPair: Tuple1[Int] = Tuple1(size._1 / 2)


  // ---------------------------------------------------------------------------
  //    Offset lookup.
  // ---------------------------------------------------------------------------
  /*
  override def endOffsetFor(inputSize: SizeLike): Int = {
    inputSize.noValues - padding._1
  }
  */


  override def offsetOfFirstPair(inputSize: Size)
  : Int = -padding0._1 * inputSize.noChannels

  /*
  override def offsetOfFirstPairOf(inputSize: SizeLike, instanceNo: Int)
  : Int = inputSize match {
    case inputSize: Size1 => offsetOfFirstPairOf(inputSize, instanceNo)
    case inputSize: Size2 => offsetOfFirstPairOf(inputSize, instanceNo)
    case inputSize: Size3 => offsetOfFirstPairOf(inputSize, instanceNo)
  }

  def offsetOfFirstPairOf(inputSize: Size1, instanceNo: Int): Int = {
    debug_req(instanceNo >= 0)
    val a = offsetOfFirstPair(inputSize)
    val b = {
      val noInstances = noOutputsForCallback(
        inputSize.noValues, size.noValues, stride._1, padding._1
      )
      debug_req(instanceNo < noInstances)
      instanceNo * stride._1
    }
    a + b
  }

  def offsetOfFirstPairOf(inputSize: Size2, instanceNo: Int): Int = {
    debug_req(instanceNo >= 0)
    val a = offsetOfFirstPair(inputSize)
    val b = {
      val outputWidth = noOutputsForCallback(
        inputSize.width, size.noValues, stride._1, padding._1
      )
      debug_req(instanceNo < outputWidth * inputSize.height)
      val y = instanceNo / outputWidth
      val x = instanceNo % outputWidth
      x * stride._1 + y * inputSize.width
    }
    a + b
  }

  def offsetOfFirstPairOf(inputSize: Size3, instanceNo: Int)
  : Int = offsetOfFirstPairOf(
    Size2(inputSize.width, inputSize.height * inputSize.depth), instanceNo
  )
*/
  /*
  override def offsetOfFirstPairOf(localPairPosition: Tuple1[Int],
                                   inputSize:         SizeLike,
                                   baseOffset:        Int)
  : Int = offsetOfFirstPairOf(
    instanceNoOf(localPairPosition, inputSize), inputSize, baseOffset
  )
  */

  override def relativeFirstOffsetOfPairOf(inputSize:   Size,
                                           localPairNo: Int)
  : Int = {
    require(localPairNo >= 0 && localPairNo < size._1)
    localPairNo * inputSize.noChannels
  }

  override def relativeFirstOffsetOfPairOf(inputSize:         Size,
                                           localPairPosition: Tuple1[Int])
  : Int = relativeFirstOffsetOfPairOf(inputSize, localPairPosition._1)


  /*
  override def relativeOffsetOfPairOf(localPairPosition: Tuple1[Int])
  : Int = relativeOffsetOfPairOf(localPairNoOf(localPairPosition))
  */

  /*
  override def relativeOffsetOfCenterPair(inputSize: SizeLike)
  : Int = size.centerIndex
  */

  // ---------------------------------------------------------------------------
  //    Derived metrics.
  // ---------------------------------------------------------------------------
  override def inputSizeFor(noChannels: Int)
  : Size1 = Size1(size, noChannels)

  def outputSizeFor(inputSize: Size1, noMaps: Int)
  : Size1 = doOutputSizeFor(inputSize, noMaps, CartesianKernel.outputSize)

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
    case inputSize: Size1 => doOutputSizeFor(inputSize, noMaps, callback)
    case inputSize: Size2 => doOutputSizeFor(inputSize, noMaps, callback)
    case inputSize: Size3 => doOutputSizeFor(inputSize, noMaps, callback)
    case inputSize: Size4 => doOutputSizeFor(inputSize, noMaps, callback)
  }

  protected def doOutputSizeFor(inputSize: Size1,
                                noMaps:    Int,
                                callback:  (Int, Int, Int, Int, Int) => Int)
  : Size1 = Size1(
    callback(inputSize.noTuples, size._1, stride._1, padding0._1, padding1._1),
    noMaps
  )

  protected def doOutputSizeFor(inputSize: Size2,
                                noMaps:    Int,
                                callback:  (Int, Int, Int, Int, Int) => Int)
  : Size2 = Size2(
    callback(inputSize.dims._1, size._1, stride._1, padding0._1, padding1._1),
    inputSize.dims._2,
    noMaps
  )

  protected def doOutputSizeFor(inputSize: Size3,
                                noMaps:    Int,
                                callback:  (Int, Int, Int, Int, Int) => Int)
  : Size3 = Size3(
    callback(inputSize.dims._1, size._1, stride._1, padding0._1, padding1._1),
    inputSize.dims._2,
    inputSize.dims._3,
    noMaps
  )

  protected def doOutputSizeFor(inputSize: Size4,
                                noMaps:    Int,
                                callback:  (Int, Int, Int, Int, Int) => Int)
  : Size4 = Size4(
    callback(inputSize.dims._1, size._1, stride._1, padding0._1, padding1._1),
    inputSize.dims._2,
    inputSize.dims._3,
    inputSize.dims._4,
    noMaps
  )


  // ---------------------------------------------------------------------------
  //    Iteration methods.
  // ---------------------------------------------------------------------------
  /**
   * Cycle through instances.
   */
  override def foreachOutput(inputSize: Size,
                             noMaps:    Int,
                             fn:        (Int, Int, Int) => Unit)
  : Unit = {
    if (ensuresAllValid) {
      inputSize match {
        case inputSize: Size1 => doForeachOutputSafe(inputSize, noMaps, fn)
        case inputSize: Size2 => doForeachOutputSafe(inputSize, noMaps, fn)
        case inputSize: Size3 => doForeachOutputSafe(inputSize, noMaps, fn)
        case inputSize: Size4 => doForeachOutputSafe(inputSize, noMaps, fn)
        case _ => throw new IllegalArgumentException
      }
    }
    else {
      inputSize match {
        case inputSize: Size1 => doForeachOutputUnsafe(inputSize, noMaps, fn)
        case inputSize: Size2 => doForeachOutputUnsafe(inputSize, noMaps, fn)
        case inputSize: Size3 => doForeachOutputUnsafe(inputSize, noMaps, fn)
        case inputSize: Size4 => doForeachOutputUnsafe(inputSize, noMaps, fn)
        case _ => throw new IllegalArgumentException
      }
    }
  }

  protected def doForeachOutputSafe(inputSize: Size1,
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

  protected def doForeachOutputSafe(inputSize: Size2,
                                    noMaps:    Int,
                                    fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val inputSize1  = inputSize.lineSize
    val outputSize1 = outputSizeFor(inputSize1, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize1.noValues * inputSize.dims._2
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize1.noValues
      doForeachOutputSafe(inputSize1, outputSize1, i0, i1, offset, fn)
      offset += inputSize1.noValues
      i0 = i1
    }
  }

  protected def doForeachOutputSafe(inputSize: Size3,
                                    noMaps:    Int,
                                    fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val inputSize1  = inputSize.lineSize
    val outputSize1 = outputSizeFor(inputSize1, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize1.noValues * inputSize.dims._2 * inputSize.dims._3
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize1.noValues
      doForeachOutputSafe(inputSize1, outputSize1, i0, i1, offset, fn)
      offset += inputSize1.noValues
      i0 = i1
    }
  }

  protected def doForeachOutputSafe(inputSize: Size4,
                                    noMaps:    Int,
                                    fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val inputSize1  = inputSize.lineSize
    val outputSize1 = outputSizeFor(inputSize1, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize1.noValues * inputSize.dims._2 * inputSize.dims._3 * inputSize.dims._4
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize1.noValues
      doForeachOutputSafe(inputSize1, outputSize1, i0, i1, offset, fn)
      offset += inputSize1.noValues
      i0 = i1
    }
  }

  protected def doForeachOutputSafe(inputSize:  Size1,
                                    outputSize: Size1,
                                    baseIndex:  Int,
                                    endIndex:   Int,
                                    baseOffset: Int,
                                    fn:         (Int, Int, Int) => Unit)
  : Unit = {
    // Pre-compute frequently used values.
    val outStepX = outputSize.noChannels
    val inpStepX = stride._1 * inputSize.noChannels

    // Move kernel through input.
    var offset = baseOffset
    var i0     = baseIndex
    while (i0 < endIndex) {
      val i1 = i0 + outStepX
      fn(i0, i1, offset)
      offset += inpStepX
      i0 = i1
    }
  }

  protected def doForeachOutputUnsafe(inputSize: Size1,
                                      noMaps:    Int,
                                      fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val outputSize = outputSizeFor(inputSize, noMaps)
    doForeachOutputUnsafe(
      inputSize, outputSize, 0, outputSize.noValues, offsetOfFirstPair(inputSize), fn
    )
  }

  protected def doForeachOutputUnsafe(inputSize: Size2,
                                      noMaps:    Int,
                                      fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val inputSize1  = inputSize.lineSize
    val outputSize1 = outputSizeFor(inputSize1, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize1.noValues * inputSize.dims._2
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize1.noValues
      doForeachOutputUnsafe(inputSize1, outputSize1, i0, i1, offset, fn)
      offset += inputSize1.noValues
      i0 = i1
    }
  }

  protected def doForeachOutputUnsafe(inputSize: Size3,
                                      noMaps:    Int,
                                      fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val inputSize1  = inputSize.lineSize
    val outputSize1 = outputSizeFor(inputSize1, noMaps)
    var offset = 0
    val iEnd   = outputSize1.noValues * inputSize.dims._2 * inputSize.dims._3
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize1.noValues
      doForeachOutputUnsafe(inputSize1, outputSize1, i0, i1, offset, fn)
      offset += inputSize1.noValues
      i0 = i1
    }
  }

  protected def doForeachOutputUnsafe(inputSize: Size4,
                                      noMaps:    Int,
                                      fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val inputSize1  = inputSize.lineSize
    val outputSize1 = outputSizeFor(inputSize1, noMaps)
    var offset = 0
    val iEnd   = outputSize1.noValues * inputSize.dims._2 * inputSize.dims._3 * inputSize.dims._4
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize1.noValues
      doForeachOutputUnsafe(inputSize1, outputSize1, i0, i1, offset, fn)
      offset += inputSize1.noValues
      i0 = i1
    }
  }

  // TODO: Could be done faster!
  protected def doForeachOutputUnsafe(inputSize:  Size1,
                                      outputSize: Size1,
                                      baseIndex:  Int,
                                      endIndex:   Int,
                                      baseOffset: Int,
                                      fn:         (Int, Int, Int) => Unit)
  : Unit = doForeachOutputUnsafe(
    inputSize, outputSize, baseIndex, endIndex, baseOffset,
    (i0, i1, offset0, x0) => fn(i0, i1, offset0)
  )

  protected def doForeachOutputUnsafe(inputSize:  Size1,
                                      outputSize: Size1,
                                      baseIndex:  Int,
                                      endIndex:   Int,
                                      baseOffset: Int,
                                      fn:         (Int, Int, Int, Int) => Unit)
  : Unit = {
    // Pre-compute frequently used values.
    val outStepX = outputSize.noChannels
    val inpStepX = stride._1 * inputSize.noChannels

    // Move kernel through input.
    var x0     = -padding0._1
    var offset = baseOffset
    var i0     = baseIndex
    while (i0 < endIndex) {
      val i1 = i0 + outStepX
      fn(i0, i1, offset, x0)
      offset += inpStepX
      x0     += stride._1
      i0 = i1
    }
  }

  override def foreachValidPairEx(inputSize: Size,
                                  noMaps:    Int,
                                  fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    if (ensuresAllValid) {
      inputSize match {
        case inputSize: Size1 => doForeachValidPairExSafe(inputSize, noMaps, fn)
        case inputSize: Size2 => doForeachValidPairExSafe(inputSize, noMaps, fn)
        case inputSize: Size3 => doForeachValidPairExSafe(inputSize, noMaps, fn)
        case inputSize: Size4 => doForeachValidPairExSafe(inputSize, noMaps, fn)
        case _ => throw new IllegalArgumentException
      }
    }
    else {
      inputSize match {
        case inputSize: Size1 => doForeachValidPairExUnsafe(inputSize, noMaps, fn)
        case inputSize: Size2 => doForeachValidPairExUnsafe(inputSize, noMaps, fn)
        case inputSize: Size3 => doForeachValidPairExUnsafe(inputSize, noMaps, fn)
        case inputSize: Size4 => doForeachValidPairExUnsafe(inputSize, noMaps, fn)
        case _ => throw new IllegalArgumentException
      }
    }
  }

  protected def doForeachValidPairExSafe(inputSize: Size1,
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

  protected def doForeachValidPairExSafe(inputSize: Size2,
                                         noMaps:    Int,
                                         fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    val inputSize1  = inputSize.lineSize
    val outputSize1 = outputSizeFor(inputSize1, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize1.noValues * inputSize.dims._2
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize1.noValues
      doForeachValidPairExSafe(inputSize1, outputSize1, i0, i1, offset, fn)
      offset += inputSize1.noValues
      i0 = i1
    }
  }

  protected def doForeachValidPairExSafe(inputSize: Size3,
                                         noMaps:    Int,
                                         fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    val inputSize1  = inputSize.lineSize
    val outputSize1 = outputSizeFor(inputSize1, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize1.noValues * inputSize.dims._2 * inputSize.dims._3
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize1.noValues
      doForeachValidPairExSafe(inputSize1, outputSize1, i0, i1, offset, fn)
      offset += inputSize1.noValues
      i0 = i1
    }
  }

  protected def doForeachValidPairExSafe(inputSize: Size4,
                                         noMaps:    Int,
                                         fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    val inputSize1  = inputSize.lineSize
    val outputSize1 = outputSizeFor(inputSize1, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize1.noValues * inputSize.dims._2 * inputSize.dims._3 * inputSize.dims._4
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize1.noValues
      doForeachValidPairExSafe(inputSize1, outputSize1, i0, i1, offset, fn)
      offset += inputSize1.noValues
      i0 = i1
    }
  }

  protected def doForeachValidPairExSafe(inputSize:  Size1,
                                         outputSize: Size1,
                                         baseIndex:  Int,
                                         endIndex:   Int,
                                         baseOffset: Int,
                                         fn:         (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    // Pre-compute frequently used values.
    val noValuesX = inputSize.noChannels * size._1

    // Foreach outputs, foreach pair.
    doForeachOutputSafe(inputSize, outputSize, baseIndex, endIndex, baseOffset,
      (i0: Int, i1: Int, baseOffset: Int) => {
        val (fnPair, fnPost) = fn(i0, i1, baseOffset)

        // Cycle through kernel dimensions.
        var offset0 = baseOffset
        var j0      = 0
        while (j0 < noValuesX) {
          val j1      = j0      + inputSize.noChannels
          val offset1 = offset0 + inputSize.noChannels
          fnPair(j0, j1, offset0, offset1)
          offset0 = offset1
          j0      = j1
        }

        // Call post.
        fnPost()
      }
    )
  }

  protected def doForeachValidPairExUnsafe(inputSize: Size1,
                                           noMaps:    Int,
                                           fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit =  {
    val outputSize = outputSizeFor(inputSize, noMaps)
    doForeachValidPairExUnsafe(
      inputSize, outputSize, 0, outputSize.noValues, offsetOfFirstPair(inputSize), fn
    )
  }

  protected def doForeachValidPairExUnsafe(inputSize: Size2,
                                           noMaps:    Int,
                                           fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    val inputSize1  = inputSize.lineSize
    val outputSize1 = outputSizeFor(inputSize1, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize1.noValues * inputSize.dims._2
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize1.noValues
      doForeachValidPairExUnsafe(inputSize1, outputSize1, i0, i1, offset, fn)
      offset += inputSize1.noValues
      i0 = i1
    }
  }

  protected def doForeachValidPairExUnsafe(inputSize: Size3,
                                           noMaps:    Int,
                                           fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    val inputSize1  = inputSize.lineSize
    val outputSize1 = outputSizeFor(inputSize1, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize1.noValues * inputSize.dims._2 * inputSize.dims._3
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize1.noValues
      doForeachValidPairExUnsafe(inputSize1, outputSize1, i0, i1, offset, fn)
      offset += inputSize1.noValues
      i0 = i1
    }
  }

  protected def doForeachValidPairExUnsafe(inputSize: Size4,
                                           noMaps:    Int,
                                           fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    val inputSize1  = inputSize.lineSize
    val outputSize1 = outputSizeFor(inputSize1, noMaps)
    var offset = offsetOfFirstPair(inputSize)
    val iEnd   = outputSize1.noValues * inputSize.dims._2 * inputSize.dims._3 * inputSize.dims._4
    var i0     = 0
    while (i0 < iEnd) {
      val i1 = i0 + outputSize1.noValues
      doForeachValidPairExUnsafe(inputSize1, outputSize1, i0, i1, offset, fn)
      offset += inputSize1.noValues
      i0 = i1
    }
  }

  protected def doForeachValidPairExUnsafe(inputSize:  Size1,
                                           outputSize: Size1,
                                           baseIndex:  Int,
                                           endIndex:   Int,
                                           baseOffset: Int,
                                           fn:         (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    // Pre-compute frequently used values.
    val maxX = inputSize.noTuples + Math.min(padding1._1, 0)

    // Foreach outputs, foreach pair.
    doForeachOutputUnsafe(inputSize, outputSize, baseIndex, endIndex, baseOffset,
      (i0: Int, i1: Int, baseOffset: Int, x0: Int) => {
        val (fnPair, fnPost) = fn(i0, i1, baseOffset)

        // TODO: Could be done slightly faster! (Call safe method if safe!)
        val x1 = x0 + size._1

        // Cycle through kernel dimensions.
        var offset0 = baseOffset
        var j0      = 0
        cfor(x0)(_ < x1, _ + 1)(x => {
          val j1      = j0      + inputSize.noChannels
          val offset1 = offset0 + inputSize.noChannels
          // TODO: Could do this slightly faster!
          if (x >= 0 && x < maxX) {
            fnPair(j0, j1, offset0, offset1)
          }
          offset0 = offset1
          j0      = j1
        })

        // Call post.
        fnPost()
      }
    )
  }

  /*
  def foreachInstance(fn: (Int, Int) => Unit, inputSize: SpatialSize)
  : SpatialSize = {
    val outputSize = outputSizeFor(inputSize)
    // Pre-compute frequently used values.
    val yGap = inputSize.width - stride._1 * outputSize.width

    // Cycle through rows, then instances.
    var offset = offsetOfFirstPair(inputSize)
    var i = 0
    while (i < outputSize.noValues) {
      val nextGapY = i + outputSize.width
      while (i < nextGapY) {
        fn(i, offset)
        offset += stride._1
        i      += 1
      }
      offset += yGap
    }
    outputSize
  }

  def foreachInstance(fn: (Int, Int) => Unit, inputSize: VolumetricSize)
  : SpatialSize = {
    val outputSize = outputSizeFor(inputSize)
    // Pre-compute frequently used values.
    val yGap = inputSize.width - stride._1 * outputSize.width

    // Cycle through planes, then rows, then, instances.
    var offset = offsetOfFirstPair(inputSize)
    var i = 0
    while (i < outputSize.noValues) {
      var y = 0
      while (y < outputSize.height) {
        var x = 0
        while (x < outputSize.width) {
          fn(i, offset)
          i      += 1
          offset += stride._1
          x      += 1
        }
        offset += yGap
        y      += 1
      }
    }
    outputSize
  }

  protected def foreachInstanceUnsafe(fn:         (Int, Int, Int) => Unit,
                                      outputSize: GenericSize,
                                      baseOffset: Int,
                                      endOffset:  Int)
  : GenericSize = {
    var offset = offsetOfFirstPair(baseOffset)

    // Cycle through instances.
    var i = 0
    while (i < outputSize.noValues) {
      val offset0 = Math.min(offset, baseOffset)
      val offset1 = Math.max(offset, endOffset)
      fn(i, offset0, offset1)
      offset += stride._1
      i      += 1
    }

    outputSize
  }

  override def foreachValidPairEx(fn:         (Int, Int) => ((Int, Int) => Unit, () => Unit),
                                  inputSize:  SizeLike,
                                  baseOffset: Int)
  : SizeLike = {
    if (padding._1 > 0) {

    }

    val outputSize = outputSizeFor(inputSize)
    if (outputSize.noValues == noValidInstancesFor(inputSize)) {
      foreachInstance(
        (i: Int, offset0: Int) => {
          val (fnPair, fnPost) = fn(i, offset0)
          // Cycle through kernel dimensions.
          var x = 0
          while (x < size.noValues) {
            fnPair(x, offset0 + x)
            x += 1
          }
          fnPost()
        },
        inputSize
      )
    }
    else {
      inputSize match {
        case inputSize: GenericSize =>
          val endOffset = endOffsetFor(inputSize, baseOffset)
          foreachInstance(
            (i: Int, offset0: Int) => {
              val (fnPair, fnPost) = fn(i, offset0)
              // Cycle through kernel dimensions.
              val offset1 = Math.min(offset0 + size.noValues, endOffset)
              var offset  = Math.max(offset0, baseOffset)
              val xDiff   = offset - offset0
              while (offset < offset1) {
                fn(offset - xDiff, offset)
                offset += 1
              }
              /*
                  // Use simplified version if safe.
                  if (offset0 >= baseOffset && offset1 <= endOffset) {
                    foreachPairCallbackSafe(index, offset0, fn)
                  }
                  else {
                    // Cycle through kernel dimensions.
                    var offset = offset0
                    var j      = 0
                    while (j < size) {
                      if (offset >= baseOffset && offset < endOffset) {
                        fn(j, offset)
                      }
                      offset += 1
                      j      += 1
                    }
                  }*/
              fnPost()
            },
            inputSize,
            baseOffset
          )
      }
    }
  }
*/
  /*
  override def foreachValidPairOf(instanceNo: Int,
                                  fn:         (Int, Int) => Unit,
                                  inputSize:  SizeLike,
                                  baseOffset: Int)
  : Unit = {
    val offset = offsetOfFirstPairOf(instanceNo, inputSize, baseOffset)
    if (noInstancesFor(inputSize) == noValidInstancesFor(inputSize)) {
      foreachPairCallbackSafe(offset, fn)
    }
    else {
      foreachPairCallbackUnsafe(offset, fn, offset0)
    }
  }
  */

}

/*
final class TemporalKernelBuilder
  extends CartesianKernelBuilder[Tuple1[Int]] {

  //override val inputSize: Tuple1[Int] = Tuple1(noInputs)

  //override val outputSize: Tuple1[Int] = Tuple1(noInstances)

  //override val size: Tuple1[Int] = Tuple1(noValues)

  //override val origin: Tuple1[Int] = Tuple1(-padding._1)


  // ---------------------------------------------------------------------------
  //    Position conversion related.
  // ---------------------------------------------------------------------------
  /*
  override def instanceNoOf(instancePos: Tuple1[Int], inputSize: SizeLike)
  : Int = {
    debug_req(instancePos._1 < noInstancesFor(inputSize))
    instancePos._1
  }
  */

  /*
  override def positionOf(index: Int): Tuple1[Int] = Tuple1(index)

  override def originOf(index: Int)
  : Tuple1[Int] = Tuple1(origin._1 + stride._1 * index)

  override def originOf(position: Tuple1[Int])
  : Tuple1[Int] = originOf(indexOf(position))

  override def destinationOf(index: Int)
  : Tuple1[Int] = Tuple1(originOf(index)._1 + noValues)

  override def destinationOf(position: Tuple1[Int])
  : Tuple1[Int] = destinationOf(indexOf(position))
  */

}
*/

object Kernel1 {

  final val one: Kernel1 = apply(1)

  final def apply(noValues: Int)
  : Kernel1 = apply(Tuple1(noValues))

  final def apply(noValues: Int, stride: Int)
  : Kernel1 = apply(Tuple1(noValues), Tuple1(stride))

  final def apply(noValues: Int, stride: Int, padding: Int)
  : Kernel1 = apply(
    Tuple1(noValues), Tuple1(stride), Tuple1(padding)
  )

  final def apply(noValues: Int, stride: Int, padding0: Int, padding1: Int)
  : Kernel1 = apply(
    Tuple1(noValues), Tuple1(stride), Tuple1(padding0), Tuple1(padding1)
  )

  final def apply(size: Tuple1[Int])
  : Kernel1 = apply(size, Tuple1(1))

  final def apply(size: Tuple1[Int], stride: Tuple1[Int])
  : Kernel1 = apply(size, stride, Tuple1(0))

  final def apply(size:    Tuple1[Int],
                  stride:  Tuple1[Int],
                  padding: Tuple1[Int])
  : Kernel1 = apply(size, stride, padding, padding)

  final def apply(size:     Tuple1[Int],
                  stride:   Tuple1[Int],
                  padding0: Tuple1[Int],
                  padding1: Tuple1[Int])
  : Kernel1 = new Kernel1(size, stride, padding0, padding1)

  final def centered(noValues: Int)
  : Kernel1 = centered(Tuple1(noValues))

  final def centered(noValues: Int, stride: Int)
  : Kernel1 = centered(Tuple1(noValues), Tuple1(stride))

  final def centered(size: Tuple1[Int])
  : Kernel1 = apply(size, Tuple1(1))

  final def centered(size: Tuple1[Int], stride: Tuple1[Int])
  : Kernel1 = apply(
    size, stride, Tuple1(size._1 / 2), Tuple1((size._1 - 1) / 2)
  )

}
