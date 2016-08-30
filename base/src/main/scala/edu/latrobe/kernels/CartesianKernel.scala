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

trait CartesianKernel[T <: Product with Serializable]
  extends Kernel {

  /**
    * Frequently used. Must override with constructor arg!
    */
  def size: T

  /**
    * Frequently used. Must override with constructor arg!
    */
  def stride: T

  /**
    * Frequently used. Must override with constructor arg!
    */
  def padding0: T

  /**
    * Frequently used. Must override with constructor arg!
    */
  def padding1: T


  // ---------------------------------------------------------------------------
  //    Pair index conversion related.
  // ---------------------------------------------------------------------------
  def localPairNoOf(localPairPos: T): Int

  def localPairPositionOf(localPairNo: Int): T

  /**
    * Must override this with a def or a lazy val.
    */
  def localPairPositionOfCenterPair: T

  final override val localPairNoOfCenterPair
  : Int = localPairNoOf(localPairPositionOfCenterPair)


  // ---------------------------------------------------------------------------
  //    Offset lookup.
  // ---------------------------------------------------------------------------
  /*
  def offsetOfFirstPairOf(localPairPosition: T,
                          inputSize:         SizeLike,
                          baseOffset:        Int)
  : Int

  final def offsetOfPairOf(instanceNo:        Int,
                           localPairPosition: T,
                           inputSize:         SizeLike,
                           baseOffset:        Int)
  : Int = offsetOfPairOf(
    instanceNo, localPairNoOf(localPairPosition), inputSize, baseOffset
  )
  */

  /*
  final def offsetOfPairOf(instancePosition: T,
                           localPairNo:      Int,
                           inputSize:        SizeLike,
                           baseOffset:       Int)
  : Int = offsetOfPairOf(
    instanceNoOf(instancePosition, inputSize),
    localPairNo,
    inputSize,
    baseOffset
  )

  final def offsetOfPairOf(instancePosition:  T,
                           localPairPosition: T,
                           inputSize:         SizeLike,
                           baseOffset:        Int)
  : Int = offsetOfPairOf(
    instancePosition, localPairNoOf(localPairPosition), inputSize, baseOffset
  )
  */

  def relativeFirstOffsetOfPairOf(inputSize: Size, localPairPosition: T)
  : Int

  final override def relativeFirstOffsetOfCenterPair(inputSize: Size)
  : Int = relativeFirstOffsetOfPairOf(
    inputSize, localPairPositionOfCenterPair
  )

  def relativeOffsetsOfPairOf(inputSize: Size, localPairPosition: T)
  : Range = {
    val offset0 = relativeFirstOffsetOfPairOf(inputSize, localPairPosition)
    offset0 until offset0 + inputSize.noChannels
  }


  // ---------------------------------------------------------------------------
  //    Derived metrics.
  // ---------------------------------------------------------------------------
  final override def outputSizeFor(inputSize: Size, noMaps: Int)
  : Size = doOutputSizeFor(inputSize, noMaps, CartesianKernel.outputSize)

  final override def fullCoverageOutputSizeFor(inputSize: Size, noMaps: Int)
  : Size = doOutputSizeFor(
    inputSize, noMaps, CartesianKernel.fullCoverageOutputSize
  )

  protected def doOutputSizeFor(inputSize: Size,
                                noMaps:    Int,
                                callback:  (Int, Int, Int, Int, Int) => Int)
  : Size

}

object CartesianKernel {

  final def isCentered(noValues: Int,
                       padding0: Int,
                       padding1: Int)
  : Boolean = {
    if (noValues / 2 != padding0) {
      false
    }
    else if (noValues % 2 == 0) {
      padding0 == padding1 + 1

    }
    else {
      padding0 == padding1
    }
  }

  final def outputSize(noInputs: Int,
                       noValues: Int,
                       stride:   Int,
                       padding0: Int,
                       padding1: Int)
  : Int = {
    val tmp = noInputs + padding0 + padding1
    if (tmp <= 0) {
      0
    }
    else if (stride == 0) {
      if (noValues <= noInputs) 1 else 0
    }
    else if (stride >= noValues) {
      val a = tmp / stride
      val b = if (tmp % stride >= noValues) 1 else 0
      a + b
    }
    else if (tmp >= noValues) {
      (tmp - noValues) / stride + 1
    }
    else {
      0
    }
  }

  final def fullCoverageOutputSize(noInputs: Int,
                                   noValues: Int,
                                   stride:   Int,
                                   padding0: Int,
                                   padding1: Int)
  : Int = {
    if (stride > 0) {
      outputSize(
        noInputs,
        noValues,
        stride,
        Math.min(padding0, 0),
        Math.min(padding1, 0)
      )
    }
    else {
      val p0 = padding0 <= 0
      val p1 = noValues - padding0 <= noInputs + Math.min(padding1, 0)
      if (p0 && p1) 1 else 0
    }
  }

}
