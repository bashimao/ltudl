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

final class Kernel4(override val size:     (Int, Int, Int, Int),
                    override val stride:   (Int, Int, Int, Int),
                    override val padding0: (Int, Int, Int, Int),
                    override val padding1: (Int, Int, Int, Int))
  extends Kernel(size._1 * size._2 * size._3 * size._4)
    with CartesianKernel[(Int, Int, Int, Int)] {
  require(size._1 > 0)
  require(size._2 > 0)
  require(size._3 > 0)
  require(size._4 > 0)
  require(stride._1 >= 0)
  require(stride._2 >= 0)
  require(stride._3 >= 0)
  require(stride._4 >= 0)
  require(padding0._1 < size._1 && padding1._1 < size._1)
  require(padding0._2 < size._2 && padding1._2 < size._2)
  require(padding0._3 < size._3 && padding1._3 < size._3)
  require(padding0._4 < size._4 && padding1._4 < size._4)

  override def toString
  : String = s"Kernel4[$size, $stride, $padding0, $padding1]"

  override def canEqual(that: Any): Boolean = that.isInstanceOf[Kernel4]

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
    case other: Kernel4 =>
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
    val w = padding0._4 <= 0 && padding1._4 <= 0
    x && y && z && w
  }

  override def isCentered: Boolean = {
    val x = CartesianKernel.isCentered(size._1, padding0._1, padding1._1)
    val y = CartesianKernel.isCentered(size._2, padding0._2, padding1._2)
    val z = CartesianKernel.isCentered(size._3, padding0._3, padding1._3)
    val w = CartesianKernel.isCentered(size._4, padding0._4, padding1._4)
    x && y && z && w
  }

  override def hasUnitStride: Boolean = {
    stride._1 == 1 && stride._2 == 1 && stride._3 == 1 && stride._4 == 1
  }

  val noValuesPerPlane: Int = size._1 * size._2

  val noValuesPerBox: Int = size._1 * size._2 * size._3


  // ---------------------------------------------------------------------------
  //     Pair index conversion related.
  // ---------------------------------------------------------------------------
  override def localPairNoOf(localPairPos: (Int, Int, Int, Int)): Int = {
    require(localPairPos._1 >= 0 && localPairPos._1 < size._1)
    require(localPairPos._2 >= 0 && localPairPos._2 < size._2)
    require(localPairPos._3 >= 0 && localPairPos._3 < size._3)
    require(localPairPos._4 >= 0 && localPairPos._4 < size._4)
    val x = localPairPos._1
    val y = localPairPos._2 * size._1
    val z = localPairPos._3 * noValuesPerPlane
    val w = localPairPos._4 * noValuesPerBox
    x + y + z + w
  }

  override def localPairPositionOf(localPairNo: Int): (Int, Int, Int, Int) = {
    require(localPairNo >= 0 && localPairNo < noValues)
    val w    = localPairNo / noValuesPerBox
    val wRem = localPairNo % noValuesPerBox
    val z    = wRem / noValuesPerPlane
    val zRem = wRem % noValuesPerPlane
    val y    = zRem / size._1
    val x    = zRem % size._1
    (x, y, z, w)
  }

  override def localPairPositionOfCenterPair
  : (Int, Int, Int, Int) = (size._1 / 2, size._2 / 2, size._3 / 2, size._4 / 2)


  // ---------------------------------------------------------------------------
  //    Offset lookup.
  // ---------------------------------------------------------------------------
  override def offsetOfFirstPair(inputSize: Size): Int = inputSize match {
    case inputSize: Size4 => offsetOfFirstPair(inputSize)
  }

  def offsetOfFirstPair(inputSize: Size4): Int = {
    val inpStride = inputSize.stride
    val x = padding0._1 * inpStride._1
    val y = padding0._2 * inpStride._2
    val z = padding0._3 * inpStride._3
    val w = padding0._4 * inpStride._4
    -x - y - z - w
  }

  override def relativeFirstOffsetOfPairOf(inputSize: Size, localPairNo: Int)
  : Int = inputSize match {
    case inputSize: Size4 => relativeOffsetOfPairOf(inputSize, localPairNo)
  }

  def relativeOffsetOfPairOf(inputSize: Size4, localPairNo: Int)
  : Int = relativeOffsetOfPairOf(inputSize, localPairPositionOf(localPairNo))

  override def relativeFirstOffsetOfPairOf(inputSize:         Size,
                                           localPairPosition: (Int, Int, Int, Int))
  : Int = inputSize match {
    case inputSize: Size4 =>
      relativeOffsetOfPairOf(inputSize, localPairPosition)
  }

  def relativeOffsetOfPairOf(inputSize: Size4, localPairPosition: (Int, Int, Int, Int))
  : Int = {
    require(localPairPosition._1 >= 0 && localPairPosition._1 < size._1)
    require(localPairPosition._2 >= 0 && localPairPosition._2 < size._2)
    require(localPairPosition._3 >= 0 && localPairPosition._3 < size._3)
    require(localPairPosition._4 >= 0 && localPairPosition._4 < size._4)
    val inpStride = inputSize.stride
    val x = localPairPosition._1 * inpStride._1
    val y = localPairPosition._2 * inpStride._2
    val z = localPairPosition._3 * inpStride._3
    val w = localPairPosition._4 * inpStride._4
    x + y + z + w
  }


  // ---------------------------------------------------------------------------
  //    Derived metrics.
  // ---------------------------------------------------------------------------
  override def inputSizeFor(noChannels: Int)
  : Size4 = Size4(size, noChannels)

  def outputSizeFor(inputSize: Size4, noMaps: Int)
  : Size4 = outputSizeFor(inputSize, noMaps, CartesianKernel.outputSize)

  override protected def doOutputSizeFor(inputSize: Size,
                                         noMaps:    Int,
                                         callback:  (Int, Int, Int, Int, Int) => Int)
  : Size = inputSize match {
    case inputSize: Size4 => outputSizeFor(inputSize, noMaps, callback)
  }

  def outputSizeFor(inputSize: Size4,
                    noMaps:    Int,
                    callback:  (Int, Int, Int, Int, Int) => Int)
  : Size4 = Size4(
    callback(inputSize.dims._1, size._1, stride._1, padding0._1, padding1._1),
    callback(inputSize.dims._2, size._2, stride._2, padding0._2, padding1._2),
    callback(inputSize.dims._3, size._3, stride._3, padding0._3, padding1._3),
    callback(inputSize.dims._4, size._4, stride._4, padding0._4, padding1._4),
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
        case inputSize: Size4 => doForeachOutputSafe(inputSize, noMaps, fn)
        case _ => throw new IllegalArgumentException
      }
    }
    else {
      inputSize match {
        case inputSize: Size4 => doForeachOutputUnsafe(inputSize, noMaps, fn)
        case _ => throw new IllegalArgumentException
      }
    }
  }

  protected def doForeachOutputSafe(inputSize: Size4,
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

  protected def doForeachOutputSafe(inputSize:  Size4,
                                    outputSize: Size4,
                                    baseIndex:  Int,
                                    endIndex:   Int,
                                    baseOffset: Int,
                                    fn:         (Int, Int, Int) => Unit)
  : Unit = {
    // Pre-compute frequently used values.
    val outStep = outputSize.stride
    val (inpStepX, gapY, gapZ, gapW) = {
      val inpStride = inputSize.stride
      val inpStepX = stride._1 * inpStride._1
      val inpStepY = stride._2 * inpStride._2
      val inpStepZ = stride._3 * inpStride._3
      val inpStepW = stride._4 * inpStride._4
      val gapY = inpStepY - inpStepX * outputSize.dims._1
      val gapZ = inpStepZ - inpStepY * outputSize.dims._2
      val gapW = inpStepW - inpStepZ * outputSize.dims._3
      (inpStepX, gapY, gapZ, gapW)
    }

    // Move kernel through input.
    var offset = baseOffset
    var i0      = baseIndex
    while (i0 < endIndex) {
      val nextGapW = i0 + outStep._4
      while (i0 < nextGapW) {
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
      offset += gapW
    }
  }

  protected def doForeachOutputUnsafe(inputSize: Size4,
                                      noMaps:    Int,
                                      fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val outputSize = outputSizeFor(inputSize, noMaps)
    doForeachOutputUnsafe(
      inputSize, outputSize, 0, outputSize.noValues, offsetOfFirstPair(inputSize), fn
    )
  }

  // TODO: Could be done faster!
  protected def doForeachOutputUnsafe(inputSize:  Size4,
                                      outputSize: Size4,
                                      baseIndex:  Int,
                                      endIndex:   Int,
                                      baseOffset: Int,
                                      fn:         (Int, Int, Int) => Unit)
  : Unit  = doForeachOutputUnsafe(
    inputSize, outputSize, baseIndex, endIndex, baseOffset,
    (i0, i1, offset0, x0, y0, z0, w0) => fn(i0, i1, offset0)
  )

  protected def doForeachOutputUnsafe(inputSize:  Size4,
                                      outputSize: Size4,
                                      baseIndex:  Int,
                                      endIndex:   Int,
                                      baseOffset: Int,
                                      fn:         (Int, Int, Int, Int, Int, Int, Int) => Unit)
  : Unit = {
    // Pre-compute frequently used values.
    val outStep = outputSize.stride
    val (inpStepX, gapY, gapZ, gapW) = {
      val inpStride = inputSize.stride
      val inpStepX = stride._1 * inpStride._1
      val inpStepY = stride._2 * inpStride._2
      val inpStepZ = stride._3 * inpStride._3
      val inpStepW = stride._4 * inpStride._4
      val gapY = inpStepY - inpStepX * outputSize.dims._1
      val gapZ = inpStepZ - inpStepY * outputSize.dims._2
      val gapW = inpStepW - inpStepZ * outputSize.dims._3
      (inpStepX, gapY, gapZ, gapW)
    }

    // Move kernel through input.
    var w0     = -padding0._4
    var offset = baseOffset
    var i0     = baseIndex
    while (i0 < endIndex) {
      var z0       = -padding0._3
      val nextGapW = i0 + outStep._4
      while (i0 < nextGapW) {
        var y0       = -padding0._2
        val nextGapZ = i0 + outStep._3
        while (i0 < nextGapZ) {
          var x0       = -padding0._1
          val nextGapY = i0 + outStep._2
          while (i0 < nextGapY) {
            val i1 = i0 + outStep._1
            fn(i0, i1, offset, x0, y0, z0, w0)
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
      offset += gapW
      w0     += stride._4
    }
  }

  override def foreachValidPairEx(inputSize: Size,
                                  noMaps:    Int,
                                  fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    if (ensuresAllValid) {
      inputSize match {
        case inputSize: Size4 => doForeachValidPairExSafe(inputSize, noMaps, fn)
        case _ => throw new IllegalArgumentException
      }
    }
    else {
      inputSize match {
        case inputSize: Size4 => doForeachValidPairExUnsafe(inputSize, noMaps, fn)
        case _ => throw new IllegalArgumentException
      }
    }
  }

  protected def doForeachValidPairExSafe(inputSize: Size4,
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

  protected def doForeachValidPairExSafe(inputSize:  Size4,
                                         outputSize: Size4,
                                         baseIndex:  Int,
                                         endIndex:   Int,
                                         baseOffset: Int,
                                         fn:         (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    // Pre-compute frequently used values.
    val noValuesX    = size._1 * inputSize.noChannels
    val noValuesXY   = size._2 * noValuesX
    val noValuesXYZ  = size._3 * noValuesXY
    val noValuesXYZW = size._4 * noValuesXYZ
    val (gapY, gapZ, gapW) = {
      val inpStride = inputSize.stride
      val gapY = inpStride._2 - noValuesX//= size._1 * inpStride._1
      val gapZ = inpStride._3 - size._2 * inpStride._2
      val gapW = inpStride._4 - size._3 * inpStride._3
      (gapY, gapZ, gapW)
    }

    // Foreach outputs, foreach pair.
    doForeachOutputSafe(inputSize, outputSize, baseIndex, endIndex, baseOffset,
      (i0: Int, i1: Int, offset0: Int) => {
        val (fnPair, fnPost) = fn(i0, i1, baseOffset)

        // Cycle through kernel dimensions.
        var offset0 = baseOffset
        var j0      = 0
        while (j0 < noValuesXYZW) {
          val nextGapW = j0 + noValuesXYZ
          while (j0 < nextGapW) {
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
          offset0 += gapW
        }

        // Call post.
        fnPost()
      }
    )
  }

  protected def doForeachValidPairExUnsafe(inputSize: Size4,
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

  protected def doForeachValidPairExUnsafe(inputSize:  Size4,
                                           outputSize: Size4,
                                           baseIndex:  Int,
                                           endIndex:   Int,
                                           baseOffset: Int,
                                           fn:         (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit = {
    // Pre-compute frequently used values.
    val maxX = inputSize.dims._1 + Math.min(padding1._1, 0)
    val maxY = inputSize.dims._2 + Math.min(padding1._2, 0)
    val maxZ = inputSize.dims._3 + Math.min(padding1._3, 0)
    val maxW = inputSize.dims._3 + Math.min(padding1._4, 0)
    val (gapY, gapZ, gapW) = {
      val inpStride = inputSize.stride
      val gapY = inpStride._2 - size._1 * inpStride._1
      val gapZ = inpStride._3 - size._2 * inpStride._2
      val gapW = inpStride._4 - size._3 * inpStride._3
      (gapY, gapZ, gapW)
    }

    /*
    val inpStride    = inputSize.stride
    val limitL       = inpStride._1 * Math.max(padding._1, 0)
    val limitR       = inpStride._1 * Math.min(Math.max(size._1 - padding._1, 0), size._1)
    val noValuesX    = inpStride._1 * size._1
    val limitT       = noValuesX * Math.max(padding._2, 0)
    val limitB       = noValuesX * Math.min(Math.max(size._2 - padding._2, 0), size._2)
    val noValuesXY   = noValuesX * size._2
    val limitN       = noValuesXY * Math.max(padding._3, 0)
    val limitF       = noValuesXY * Math.min(Math.max(size._3 - padding._3, 0), size._3)
    val noValuesXYZ  = noValuesXY * size._3
    val limit0       = noValuesXYZ * Math.max(padding._4, 0)
    val limit1       = noValuesXYZ * Math.min(Math.max(size._4 - padding._4, 0), size._4)
    val noValuesXYZW = noValuesXYZ * size._4
    */

    // Foreach outputs, foreach pair.
    doForeachOutputUnsafe(inputSize, outputSize, baseIndex, endIndex, baseOffset,
      // TODO: Can do this slightly faster!
      (i0: Int, i1: Int, baseOffset: Int, x0: Int, y0: Int, z0: Int, w0: Int) => {
        val (fnPair, fnPost) = fn(i0, i1, baseOffset)

        // TODO: Could be done slightly faster! (Call safe method if safe!)
        val x1 = x0 + size._1
        val y1 = y0 + size._2
        val z1 = z0 + size._3
        val w1 = w0 + size._4

        /*
        // If unsafe kernel instance (left, right, top, bottom, near, far, past, late)
        val begX = if (x0 == 0) limitL else 0
        val begY = if (y0 == 0) limitT else 0
        val begZ = if (z0 == 0) limitN else 0
        val begW = if (w0 == 0) limit0 else 0
        val endX = if (x0 == outputSize.dims._1 - 1) limitR else noValuesX
        val endY = if (y0 == outputSize.dims._2 - 1) limitB else noValuesXY
        val endZ = if (z0 == outputSize.dims._3 - 1) limitF else noValuesXYZ
        val endW = if (w0 == outputSize.dims._4 - 1) limit0 else noValuesXYZW
        val remY = noValuesX   - endX
        val remZ = noValuesXY  - endY
        val remW = noValuesXYZ - endZ
        val gapY = inpStride._2 - endX // Note different gaps!
        //val gapZ = (inputSize.height - yEnd) * inputSize.width
        // TODO: Can do this slightly faster!
        val gapZ = inpStride._3 - inpStride._2 * (endY / noValuesX)
        val gapW = inpStride._4 - inpStride._3 * (endZ / noValuesXY)
        */

        // Cycle through kernel dimensions.
        var offset0 = baseOffset
        var j0      = 0
        cfor(w0)(_ < w1, _ + 1)(w => {
          cfor(z0)(_ < z1, _ + 1)(z => {
            cfor(y0)(_ < y1, _ + 1)(y => {
              cfor(x0)(_ < x1, _ + 1)(x => {
                val j1      = j0      + inputSize.noChannels
                val offset1 = offset0 + inputSize.noChannels
                // TODO: Could do this slightly faster!
                if (x >= 0 && x < maxX && y >= 0 && y < maxY && z >= 0 && z < maxZ && w >= 0 && w < maxW) {
                  fnPair(j0, j1, offset0, offset1)
                }
                offset0 = offset1
                j0      = j1
              })
              offset0 += gapY
            })
            offset0 += gapZ
          })
        })
        /*
        var offset0 = baseOffset
        var j0      = begW
        while (j0 < endW) {
          val nextGapZ = j0 + endZ
          j0 += begZ
          while (j0 < nextGapZ) {
            val nextGapY = j0 + endY
            j0 += begY
            while (j0 < nextGapY) {
              val nextGapX = j0 + endX
              j0 += begX
              while (j0 < nextGapX) {
                val j1      = j0      + inpStride._1
                val offset1 = offset0 + inpStride._1
                fnPair(j0, j1, offset0, offset1)
                offset0 = offset1
                j0      = j1
              }
              offset0 += gapY
              j0      += remY
            }
            offset0 += gapZ
            j0      += remZ
          }
          offset0 += gapW
          j0      += remW
        }
        */

        // Call post.
        fnPost()
      }
    )
  }

}

object Kernel4 {

  final def apply(size: (Int, Int, Int, Int))
  : Kernel4 = apply(size, (1, 1, 1, 1))

  final def apply(size:   (Int, Int, Int, Int),
                  stride: (Int, Int, Int, Int))
  : Kernel4 = apply(size, stride, (0, 0, 0, 0))

  final def apply(size:    (Int, Int, Int, Int),
                  stride:  (Int, Int, Int, Int),
                  padding: (Int, Int, Int, Int))
  : Kernel4 = apply(size, stride, padding, padding)

  final def apply(size:     (Int, Int, Int, Int),
                  stride:   (Int, Int, Int, Int),
                  padding0: (Int, Int, Int, Int),
                  padding1: (Int, Int, Int, Int))
  : Kernel4 = new Kernel4(size, stride, padding0, padding1)

  final def centered(size: (Int, Int, Int, Int))
  : Kernel4 = centered(size, (1, 1, 1, 1))

  final def centered(size:   (Int, Int, Int, Int),
                     stride: (Int, Int, Int, Int))
  : Kernel4 = apply(
    size,
    stride,
    (size._1 / 2, size._2 / 2, size._3 / 2, size._4 / 2),
    ((size._1 - 1) / 2, (size._2 - 1) / 2, (size._3 - 1) / 2, (size._4 - 1) / 2)
  )

}
