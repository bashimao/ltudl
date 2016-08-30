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

package edu.latrobe

/**
  * New interpretation of the kernel as a purely geometric primitive for
  * iterating over layers.
  *
  * Although this is called a kernel, it is more or less the description of how
  * a kernel with specific properties moves though a predefined space.
  *
  * Method naming scheme:
  * ...    = Not associated with an explicit item.
  * ...For = For some input of the specified size
  * ...Of  = A specific element (can be tied to a specific input size)
  */
abstract class Kernel(val noValues: Int)
  extends Serializable
    with Equatable {

  /**
    * Frequently used. Override this with a val!
    *
    * A flag that indicates whether the kernel ensures that full convolution
    * and valid convolution is the same.
 *
    * @return True if the kernel guarantees to never pad the input.
    */
  def ensuresAllValid
  : Boolean

  /**
    * One of the two flags that can be used to determine whether the kernel
    * produces exactly one output for each input.
 *
    * @return True if the kernel moves one by one in all possible dimensions.
    */
  def isCentered
  : Boolean

  /**
    * On of the two flags that can be used to determine whether the kernel
    * produces exactly one output for each input.
 *
    * @return True if the kernel's center is the inverse of its padding.
    */
  def hasUnitStride
  : Boolean


  // ---------------------------------------------------------------------------
  //    Pair number conversion related.
  // ---------------------------------------------------------------------------
  final def outputNoOf(globalPairNo: Int)
  : Int = globalPairNo / noValues

  final def localPairNoOf(globalPairNo: Int)
  : Int = globalPairNo % noValues

  /**
   * This will be used frequently. If you implement this interface, making this
   * computation fast should have priority! Best: Make it a val!
   */
  def localPairNoOfCenterPair
  : Int

  /**
   * Shorthand for retrieving th first index of the first pair belonging to that
   * instance..
 *
   * @param outputNo Index of output.
   * @return
   */
  final def globalPairNoOf(outputNo: Int)
  : Int = outputNo * noValues

  final def globalPairNoOf(outputNo: Int, localPairNo: Int)
  : Int = globalPairNoOf(outputNo) + localPairNo

  final def globalPairNoOfCenterPairOf(outputNo: Int)
  : Int = globalPairNoOf(outputNo, localPairNoOfCenterPair)


  // ---------------------------------------------------------------------------
  //    Offset lookup.
  // ---------------------------------------------------------------------------
  def offsetOfFirstPair(inputSize: Size)
  : Int
  //def endOffsetFor(inputSize: SizeLike): Int

  /*
  final def offsetOfPairOf(inputSize:   SizeLike,
                           outputNo:    Int,
                           localPairNo: Int)
  : Int = {
    val a = offsetOfFirstPairOf(inputSize, outputNo)
    val b = relativeOffsetOfPairOf(inputSize, localPairNo)
    a + b
  }
  */

  /*
  final def offsetOfPairOf(inputSize: SizeLike, globalPairNo: Int)
  : Int = offsetOfPairOf(
    inputSize, outputNoOf(globalPairNo), localPairNoOf(globalPairNo)
  )
*/

  /*
    def offsetOfFirstPairOf(inputSize: SizeLike, outputNo: Int)
    : Int

    final def offsetOfFirstPairOfLastInstanceFor(inputSize: SizeLike)
    : Int = offsetOfFirstPairOf(inputSize, noOutputsFor(inputSize) - 1)
    */
  /*
  final def offsetOfLastPairFor(inputSize: SizeLike): Int = {
    val a = offsetOfFirstPairOfLastInstanceFor(inputSize)
    val b = relativeOffsetOfPairOf(inputSize, size.noValues - 1)
    a + b
  }
  */

  /*
  final def offsetOfCenterPairOf(inputSize: SizeLike, instanceNo: Int): Int = {
    val a = offsetOfFirstPairOf(inputSize, instanceNo)
    val b = relativeOffsetOfCenterPair(inputSize)
    a + b
  }
  */

  def relativeFirstOffsetOfPairOf(inputSize: Size, localPairNo: Int)
  : Int

  def relativeFirstOffsetOfCenterPair(inputSize: Size)
  : Int

  final def relativeOffsetsOfPairOf(inputSize: Size, localPairNo: Int)
  : Range = {
    val offset0 = relativeFirstOffsetOfPairOf(inputSize, localPairNo)
    offset0 until offset0 + inputSize.noChannels
  }

  final def relativeOffsetsOfCenterPair(inputSize: Size)
  : Range = {
    val offset0 = relativeFirstOffsetOfCenterPair(inputSize)
    offset0 until offset0 + inputSize.noChannels
  }


  // ---------------------------------------------------------------------------
  //    Derived metrics.
  // ---------------------------------------------------------------------------
  /**
    * Returns the input size of a single output unit.
    */
  def inputSizeFor(noChannels: Int)
  : Size

  def outputSizeFor(inputSize: Size, noMaps: Int)
  : Size

  final def noOutputsFor(inputSize: Size, noMaps: Int)
  : Int = outputSizeFor(inputSize, noMaps).noValues

  /**
    * Size just considering outputs with full coverage.
    */
  def fullCoverageOutputSizeFor(inputSize: Size, noMaps: Int)
  : Size

  /**
    * Number of outputs with full coverage.
    */
  final def noFullCoverageOutputsFor(inputSize: Size, noMaps: Int)
  : Int = fullCoverageOutputSizeFor(inputSize, noMaps).noValues


  // ---------------------------------------------------------------------------
  //    Iteration methods.
  // ---------------------------------------------------------------------------
  def foreachOutput(inputSize: Size,
                    noMaps:    Int,
                    fn:        (Int, Int, Int) => Unit)
  : Unit

  final def foreachValidPair(inputSize: Size,
                             noMaps:    Int,
                             fn:        (Int, Int, Int) => (Int, Int, Int, Int) => Unit)
  : Unit = foreachValidPairEx(
    inputSize, noMaps, (i0, i1, offset0) => (fn(i0, i1, offset0), () => Unit)
  )

  // TODO: May need direct implementation to avoid time consuming division.
  final def foreachValidPairIndex(inputSize: Size,
                                  noMaps:    Int,
                                  fn:        (Int, Int, Int) => (Int, Int, Int) => Unit)
  : Unit = foreachValidPairIndexEx(
    inputSize, noMaps, (i0, i1, offset0) => (fn(i0, i1, offset0), () => Unit)
  )

  def foreachValidPairEx(inputSize: Size,
                         noMaps:    Int,
                         fn:        (Int, Int, Int) => ((Int, Int, Int, Int) => Unit, () => Unit))
  : Unit

  // TODO: May need direct implementation to avoid time consuming division.
  final def foreachValidPairIndexEx(inputSize: Size,
                                    noMaps:    Int,
                                    fn:        (Int, Int, Int) => ((Int, Int, Int) => Unit, () => Unit))
  : Unit = foreachValidPairEx(inputSize, noMaps, (i0, i1, offset0) => {
    val (fnPair, fnPost) = fn(i0, i1, offset0)
    ((j0, j1, offset0, offset1) => fnPair(j0 / inputSize.noChannels, offset0, offset1), fnPost)
  })

  /*
  final def foreachValidPairOf(instanceNo: Int,
                               fn:         (Int, Int) => Unit,
                               inputSize:  SizeLike)
  : SizeLike = foreachValidPairOf(instanceNo, fn, inputSize, 0)

  def foreachValidPairOf(instanceNo: Int,
                         fn:         (Int, Int) => Unit,
                         inputSize:  SizeLike,
                         baseOffset: Int)
  : SizeLike
  */


  // ---------------------------------------------------------------------------
  //    Advanced stuff.
  // ---------------------------------------------------------------------------
  /*
  /**
   * Slow! Cache result if used multiple times.
   */
  final def computeWeights: DVec = {
    val result = Array.ofDim[Real](noInputs)
    foreachValidPair((i, offset) => (j, offset) => result(offset) += Real.one, 0, 0)
    DenseVector(result)
  }

  /**
   * Slow! Cache result if used multiple times.
   */
  final def computeWeightsInv: DVec = {
    val tmp = computeWeights
    tmp.transform(Real.one / _)
    tmp
  }
  */

}

/*
abstract class KernelBuilder extends InstanceBuilder0 {

  // ---------------------------------------------------------------------------
  //    Pair number conversion related.
  // ---------------------------------------------------------------------------
  final def noPairsFor(inputSize: Size, noMaps: Int)
  : Int = noOutputsFor(inputSize, noMaps) * noValues

  //def localPairNoOfCenterPair: Int

}
*/