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

/*
/**
 * A more advanced kernel that uses multiple kernels to work on tiled inputs.
 */
trait KernelExLike extends KernelLike {

  def kernel: KernelLike

  def noTiles: Int

  final override def noInputs: Int = kernel.noInputs * noTiles

  final override def noInstances: Int = kernel.noInstances * noTiles

  final override def noValidInstances: Int = kernel.noValidInstances * noTiles

  final override def noValues: Int = kernel.noValues


  // ---------------------------------------------------------------------------
  //     Pair index conversion related.
  // ---------------------------------------------------------------------------
  final override def localPairIndexOfCenterPair
  : Int = kernel.localPairIndexOfCenterPair


  // ---------------------------------------------------------------------------
  //    Offset lookup.
  // ---------------------------------------------------------------------------
  final override def offsetOfFirst(baseOffset: Int)
  : Int = kernel.offsetOfFirst(baseOffset)

  override def offsetOf(index: Int, baseOffset: Int): Int = {
    val n        = noTiles
    val newIndex = index % n
    val offset   = baseOffset + (index / n) * kernel.noInputs
    kernel.offsetOf(newIndex, offset)
  }

  final override def relativeOffsetOfPair(localPairIndex: Int)
  : Int = kernel.relativeOffsetOfPair(localPairIndex)

  final override def relativeOffsetOfCenterPair
  : Int = kernel.relativeOffsetOfCenterPair


  // ---------------------------------------------------------------------------
  //    Iteration methods.
  // ---------------------------------------------------------------------------
  override def foreach(fn: (Int, Int) => Unit, index0: Int, offset0: Int)
  : Unit = {
    var offset = offset0
    val iEnd   = index0 + noInstances
    var i      = index0
    while (i < iEnd) {
      kernel.foreach(fn, i, offset)
      offset += kernel.noInputs
      i      += kernel.noInstances
    }
  }

  final override def foreachValidPair(fn:      (Int, Int) => (Int, Int) => Unit,
                                 index0:  Int,
                                 offset0: Int)
  : Unit = {
    var offset = offset0
    val iEnd   = index0 + noInstances
    var i      = index0
    while (i < iEnd) {
      kernel.foreachValidPair(fn, i, offset)
      offset += kernel.noInputs
      i      += kernel.noInstances
    }
  }

  final override def foreachValidPair(index:   Int,
                                 fn:      (Int, Int) => Unit,
                                 index0:  Int,
                                 offset0: Int)
  : Unit = {
    val newIndex    = index % kernel.noInstances
    val tileNo      = index / kernel.noInstances
    val tileIndex0  = tileNo * kernel.noInstances + index0
    val tileOffset0 = tileNo * kernel.noInputs + offset0
    kernel.foreachValidPair(newIndex, fn, tileIndex0, tileOffset0)
  }

}
*/

/**
 * Extends KernelLike with a logical coordinate system.
 * @tparam T Type used for expressing coordinates.
 */
/*
abstract class KernelExBuilder[T <: Serializable] extends KernelBuilder {

  // ---------------------------------------------------------------------------
  //    Pair index conversion related.
  // ---------------------------------------------------------------------------
  def localPairPositionOfCenterPair: T


  // ---------------------------------------------------------------------------
  //    Position conversion related.
  // ---------------------------------------------------------------------------
  //def instanceNoOf(instancePos: T, inputSize: SizeLike): Int

  //def positionOfFirstPairFor(inputSize: SizeLike, instanceNo: Int): T

  //def originOf(instanceNo: Int): T

  //def originOf(position: T): T

  //final def originOfLast: T = originOf(noInstances - 1)

  //def destinationOf(index: Int): T

  //def destinationOf(position: T): T

  //final def destinationOfLast: T = destinationOf(noInstances - 1)

}
*/