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

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions._
import com.google.common.util.concurrent._
import edu.latrobe.sizes._
import scala.reflect._

/**
 * Encapsulates completely materialized mini-batch.
 *
 * The backing storage is an DMat.
 *
 * @param values Value buffer of this instance.
 */
final class RealArrayTensor(override val layout: IndependentTensorLayout,
                            override val values: Array[Real])
  extends TensorEx[RealArrayTensor]
    with RealTensor
    with Serializable {
  require(layout.noValues == values.length)

  override def repr
  : RealArrayTensor = this

  override def toString
  : String = s"RealArrayTensor[$layout]"

  override def platform
  : JVM.type = JVM

  override def createSibling(newLayout: TensorLayout)
  : RealArrayTensor = RealArrayTensor.zeros(newLayout.makeIndependent)

  override def createSiblingAndClear(newLayout: TensorLayout)
  : RealArrayTensor = RealArrayTensor.zeros(newLayout.makeIndependent)

  override def copy
  : RealArrayTensor = RealArrayTensor(layout, values.clone)

  /*
  @transient
  lazy val valuesParallel: ParArray[DenseVector[Real]] = {
    val builder = ParArray.newBuilder[DenseVector[Real]]
    var i0 = 0
    while (i0 < values.length) {
      var i1 = Math.min(i0 + BLAZE_PARALLELIZATION_THRESHOLD_MAX, values.length)

      // Make sure that next segment does not become too small.
      if (values.length - i1 < BLAZE_PARALLELIZATION_THRESHOLD_MIN) {
        i1 = values.length
      }

      builder += new DenseVector(values, i0, 1, i1 - i0)

      i0 = i1
    }
    builder.result()
  }
  */

  @transient
  override lazy val valuesMatrix
  : DenseMatrix[Real] = {
    val rows = layout.size.noValues
    val cols = layout.noSamples
    new DenseMatrix(rows, cols, values)
  }

  override def valuesMatrixEx
  : DenseMatrix[Real] = valuesMatrix

  /*
  @transient
  private lazy val valueRowOffsets
  : Range = 0 until layout.size.noValues

  @transient
  private lazy val valueRowVectors
  : Array[DenseVector[Real]] = MatrixEx.majorVectors(valuesMatrix)

  @transient
  private lazy val channelMatrix
  : DenseMatrix[Real] = new DenseMatrix(
    layout.size.noChannels,
    layout.size.noTuples * layout.noSamples,
    values
  )
  */

  /*
  @transient
  private lazy val channelOffsets
  : Range = 0 until layout.size.noChannels
  */

  /*
  @transient
  private lazy val channelVectors
  : Array[DenseVector[Real]] = MatrixEx.majorVectors(channelMatrix)
  */

  /*
  @transient
  private lazy val sampleVectors
  : Array[DenseVector[Real]] = MatrixEx.minorVectors(valuesMatrix)

  @transient
  private lazy val sampleChannelVectors
  : Array[Array[DenseVector[Real]]] = {
    val noTuples   = layout.size.noTuples
    val noChannels = layout.size.noChannels
    val result     = new Array[Array[DenseVector[Real]]](layout.noSamples)
    RangeEx.mapPairs(sampleOffsets)((i, off0) => {
      val sample = new DenseMatrix(noChannels, noTuples, values, off0, noChannels)
      result(i) = MatrixEx.majorVectors(sample)
    })
    result
  }
  */

  /*
  @transient
  private lazy val sampleChannelVectors
  : Array[DenseVector[Real]] = {
    val a = ArrayEx.map(sampleChannelMatrices)(MatrixEx.majorVectors)

  }
  */

  /*
  override def valuesMatrixCopy
  : DenseMatrix[Real] = new DenseMatrix(
    layout.size.noValues, layout.noSamples, values.clone
  )
  */

  /*
  @transient
  lazy val valuesMatrixColumns
  : Array[DenseVector[Real]] = MatrixEx.columnVectors(valuesMatrix)

  @transient
  lazy val valuesMatrixColumnsParallel
  : ParArray[DenseVector[Real]] = ParArray.handoff(valuesMatrixColumns)

  @transient
  lazy val valuesMatrixColumnPairs
  : Array[(Int, DenseVector[Real])] = MatrixEx.columnVectorPairs(valuesMatrix)

  @transient
  lazy val valuesMatrixColumnPairsParallel
  : ParArray[(Int, DenseVector[Real])] = ParArray.handoff(valuesMatrixColumnPairs)
  */

  override def get(valueNo: Int)
  : Real = values(valueNo)

  override def get(result: Array[Real], offset: Int, stride: Int)
  : Unit = ArrayEx.set(
    result, offset, stride,
    values, 0,      1,
    values.length
  )

  override def put(valueNo: Int, value: Real)
  : Unit = values(valueNo) = value

  override def put(array: Array[Real], offset: Int, stride: Int)
  : Unit = ArrayEx.set(
    values, 0,      1,
    array,  offset, stride,
    values.length
  )

  override def clear()
  : Unit = ArrayEx.fill(values, Real.zero)

  override def fill(fn: () => Real, threadSafe: Boolean)
  : Unit = {
    if (threadSafe) {
      foreachChunk(
        LTU_REAL_TENSOR_CHUNK_SIZE_FOR_FILL,
        ArrayEx.fill(
          values, _, 1,
          _
        )(fn())
      )
    }
    else {
      ArrayEx.fill(
        values
      )(fn())
    }
  }


  // ---------------------------------------------------------------------------
  //    Basic operations.
  // ---------------------------------------------------------------------------
  override def reshape(newSize: Size)
  : RealArrayTensor = {
    require(newSize.noValues == layout.size.noValues)
    RealArrayTensor(
      layout.derive(newSize),
      values.clone
    )
  }

  override def apply(index: Int)
  : RealArrayTensor = RealArrayTensor.derive(
    layout.size,
    valuesMatrix(::, index)
  )

  override def apply(indices: Range)
  : RealArrayTensor = RealArrayTensor.derive(
    layout.size,
    valuesMatrix(::, indices)
  )

  override def splitSamples
  : Array[Tensor] = MatrixEx.mapColumnVectors(
    valuesMatrix
  )(RealArrayTensor.derive(layout.size, _))

  override def concat(other: Tensor)
  : RealArrayTensor = {
    val newLayout = layout.concat(other.layout)

    val result = RealArrayTensor.zeros(newLayout)
    ArrayEx.concatEx(
      result.values,
      values,
      other.values
    )
    result
  }

  override def concat[T <: Tensor](others: Array[T])
  : RealArrayTensor = {
    val newLayout = ArrayEx.foldLeft(
      layout,
      others
    )((res, tensor) => res.concat(tensor.layout))

    val otherValues = ArrayEx.map(others)(_.values)
    val result      = RealArrayTensor.zeros(newLayout)
    ArrayEx.concatEx(
      result.values,
      values,
      otherValues
    )
    result
  }

  override protected def doSet(value: Real)
  : Unit = {
    ArrayEx.fill(
      values,
      value
    )
  }

  override protected def doSet(other: Tensor)
  : Unit = {
    ArrayEx.set(
      values,
      other.values
    )
  }

  override protected def doSet(other: Tensor, beta: Real)
  : Unit = {
    val otherValues = other.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SET,
      (off0, length0) => {
        ArrayEx.set(
          values,      off0, 1,
          beta,
          otherValues, off0, 1,
          length0
        )
      }
    )
  }

  override protected def doSet(other0: Tensor, other1: Tensor)
  : Unit = {
    val other0Values = other0.values
    val other1Values = other1.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SET,
      (off0, length0) => {
        ArrayEx.set(
          values,       off0, 1,
          other0Values, off0, 1,
          other1Values, off0, 1,
          length0
        )
      }
    )
  }

  override protected def doSet(other0: Tensor, other1: Tensor, beta: Real)
  : Unit = {
    val other0Values = other0.values
    val other1Values = other1.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SET,
      (off0, length0) => {
        ArrayEx.set(
          values,       off0, 1,
          beta,
          other0Values, off0, 1,
          other1Values, off0, 1,
          length0
        )
      }
    )
  }

  override protected def doAdd(other: Tensor)
  : Unit = {
    val otherValues = other.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ADD,
      (off0, length0) => {
        ArrayEx.add(
          values,      off0, 1,
          otherValues, off0, 1,
          length0
        )
      }
    )
  }

  override protected def doAdd(value: Real)
  : Unit = {
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ADD,
      (off0, length0) => {
        ArrayEx.add(
          values, off0, 1,
          value,
          length0
        )
      }
    )
  }

  override protected def doAdd(alpha: Real,
                               other: Tensor)
  : Unit = {
    val otherValues = other.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ADD,
      (off0, length0) => {
        ArrayEx.add(
          alpha,
          values,      off0, 1,
          otherValues, off0, 1,
          length0
        )
      }
    )
  }

  override protected def doAdd(alpha: Real,
                               other: Tensor, beta: Real)
  : Unit = {
    val otherValues = other.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ADD,
      (off0, length0) => {
        ArrayEx.add(
          alpha,
          values,      off0, 1,
          beta,
          otherValues, off0, 1,
          length0
        )
      }
    )
  }

  override protected def doAdd(other: Tensor, beta: Real)
  : Unit = {
    val otherValues = other.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ADD,
      (off0, length0) => {
        ArrayEx.add(
          values,      off0, 1,
          beta,
          otherValues, off0, 1,
          length0
        )
      }
    )
  }

  override protected def doAdd(other0: Tensor, other1: Tensor)
  : Unit = {
    val other0Values = other0.values
    val other1Values = other1.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ADD,
      (off0, length0) => {
        ArrayEx.add(
          values,       off0, 1,
          other0Values, off0, 1,
          other1Values, off0, 1,
          length0
        )
      }
    )
  }

  override protected def doAdd(alpha: Real,
                               other0: Tensor, other1: Tensor)
  : Unit = {
    val other0Values = other0.values
    val other1Values = other1.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ADD,
      (off0, length0) => {
        ArrayEx.add(
          alpha,
          values,       off0, 1,
          other0Values, off0, 1,
          other1Values, off0, 1,
          length0
        )
      }
    )
  }

  override protected def doAdd(other0: Tensor, other1: Tensor, beta: Real)
  : Unit = {
    val other0Values = other0.values
    val other1Values = other1.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ADD,
      (off0, length0) => {
        ArrayEx.add(
          values,       off0, 1,
          beta,
          other0Values, off0, 1,
          other1Values, off0, 1,
          length0
        )
      }
    )
  }

  override protected def doAdd(alpha: Real,
                               other0: Tensor, other1: Tensor, beta: Real)
  : Unit = {
    val other0Values = other0.values
    val other1Values = other1.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ADD,
      (off0, length0) => {
        ArrayEx.add(
          alpha,
          values,       off0, 1,
          beta,
          other0Values, off0, 1,
          other1Values, off0, 1,
          length0
        )
      }
    )
  }

  override protected def doSubtract(other: Tensor)
  : Unit = {
    val otherValues = other.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ADD,
      (off0, length0) => {
        ArrayEx.subtract(
          values,      off0, 1,
          otherValues, off0, 1,
          length0
        )
      }
    )
  }

  override def subtractR(value: Real)
  : Unit = {
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ADD,
      (off0, length0) => {
        ArrayEx.subtract(
          value,
          values, off0, 1,
          length0
        )
      }
    )
  }

  override protected def doMultiply(value: Real)
  : Unit = {
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MULTIPLY,
      (off0, length0) => {
        ArrayEx.multiply(
          values, off0, 1,
          value,
          length0
        )
      }
    )
  }

  override protected def doMultiply(other: Tensor)
  : Unit = {
    val otherValues = other.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MULTIPLY,
      (off0, length0) => {
        ArrayEx.multiply(
          values,      off0, 1,
          otherValues, off0, 1,
          length0
        )
      }
    )
  }

  override protected def doMultiply(other: Tensor, beta: Real)
  : Unit = {
    val otherValues = other.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MULTIPLY,
      (off0, length0) => {
        ArrayEx.multiply(
          values,      off0, 1,
          otherValues, off0, 1,
          beta,
          length0
        )
      }
    )
  }

  override protected def doDivide(other: Tensor)
  : Unit = {
    val otherValues = other.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_DIVIDE,
      (off0, length0) => {
        ArrayEx.divide(
          values,      off0, 1,
          otherValues, off0, 1,
          length0
        )
      }
    )
  }

  override protected def doDivide(epsilon0: Real,
                                  other:    Tensor)
  : Unit = {
    val otherValues = other.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_DIVIDE,
      (off0, length0) => {
        ArrayEx.divide(
          epsilon0,
          values,      off0, 1,
          otherValues, off0, 1,
          length0
        )
      }
    )
  }

  override protected def doDivide(other: Tensor, epsilon1: Real)
  : Unit = {
    val otherValues = other.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_DIVIDE,
      (off0, length0) => {
        ArrayEx.divide(
          values,      off0, 1,
          epsilon1,
          otherValues, off0, 1,
          length0
        )
      }
    )
  }

  override protected def doDivide(epsilon0: Real,
                                  other: Tensor, epsilon1: Real)
  : Unit = {
    val otherValues = other.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_DIVIDE,
      (off0, length0) => {
        ArrayEx.divide(
          epsilon0,
          values,      off0, 1,
          epsilon1,
          otherValues, off0, 1,
          length0
        )
      }
    )
  }

  override protected def doDot(other: Tensor)
  : Real = {
    val result      = new AtomicDouble()
    val otherValues = other.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_DOT,
      (off0, length0) => {
        val tmp = ArrayEx.dot(
          values,      off0, 1,
          otherValues, off0, 1,
          length0
        )
        result.addAndGet(DoubleEx(tmp))
      }
    )
    Real(result.doubleValue)
  }

  override protected def doDivideR(value: Real)
  : Unit = {
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MULTIPLY,
      (off0, length0) => {
        ArrayEx.divide(
          value,
          values, off0, 1,
          length0
        )
      }
    )
  }

  def lerp(value: Real, t: Real)
  : Unit = {
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_LERP,
      ArrayEx.lerp(
        values, _, 1,
        value,
        _,
        t
      )
    )
  }

  override protected def doLerp(other: Tensor, t: Real)
  : Unit = {
    val otherValues = other.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_LERP,
      (off0, length0) => {
        ArrayEx.lerp(
          values,      off0, 1,
          otherValues, off0, 1,
          length0,
          t
        )
      }
    )
  }

  override protected def doLerp(other0: Tensor, other1: Tensor, t: Real)
  : Unit = {
    val other0Values = other0.values
    val other1Values = other1.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_LERP,
      (off0, length0) => {
        ArrayEx.lerp(
          values,       off0, 1,
          other0Values, off0, 1,
          other1Values, off0, 1,
          length0,
          t
        )
      }
    )
  }


  // ---------------------------------------------------------------------------
  //    Fancy operations.
  // ---------------------------------------------------------------------------
  override def abs()
  : Unit = {
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ABS,
      ArrayEx.abs(
        values, _, 1,
        _
      )
    )
  }

  override def l1Norm(epsilon: Double)
  : Real = {
    val result = new AtomicDouble()
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_L1_NORM,
      (off0, length0) => {
        val tmp = ArrayEx.l1Norm(
          values, off0, 1,
          length0,
          epsilon
        )
        result.addAndGet(DoubleEx(tmp))
      }
    )
    Real(result.doubleValue)
  }

  override def l2Norm(epsilon: Double)
  : Real = Real(Math.sqrt(l2NormSq + epsilon))

  override def l2NormSq
  : Real = {
    val result = new AtomicDouble()
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_L2_NORM,
      (off0, length0) => {
        val tmp = ArrayEx.l2NormSq(
          values, off0, 1,
          length0
        )
        result.addAndGet(DoubleEx(tmp))
      }
    )
    Real(result.doubleValue)
  }

  override def mean
  : Real = sum / layout.noValues

  override def max()
  : Real = {
    val tmp = mapChunks(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MAX,
      ArrayEx.max(
        values, _, 1,
        _
      )
    )
    ArrayEx.max(tmp)
  }

  override def max(other: Tensor)
  : Unit = {
    require(layout == other.layout)

    val otherValues = other.values

    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MAX,
      (off0, length0) => {
        ArrayEx.max(
          values,      off0, 1,
          otherValues, off0, 1,
          length0
        )
      }
    )
  }

  override def maxAbs()
  : Real = {
    reduceChunks(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MAX
    )(
      ArrayEx.maxAbs(
        values, _, 1,
        _
      )
    )(Math.max)
  }

  override protected def doMaxByAbs(other: Tensor)
  : Unit = {
    val otherValues = other.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MAX,
      (off0, length0) => {
        ArrayEx.maxByAbs(
          values,      off0, 1,
          otherValues, off0, 1,
          length0
        )
      }
    )
  }

  override def min()
  : Real = {
    val tmp = mapChunks(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MIN,
      ArrayEx.min(
        values, _, 1,
        _
      )
    )
    ArrayEx.min(tmp)
  }

  override def min(other: Tensor)
  : Unit = {
    require(layout == other.layout)

    val otherValues = other.values
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MIN,
      (off0, length0) => {
        ArrayEx.min(
          values,      off0, 1,
          otherValues, off0, 1,
          length0
        )
      }
    )
  }

  override def sign()
  : Unit = {
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SIGN,
      (off0, length0) => {
        ArrayEx.sign(
          values, off0, 1,
          length0
        )
      }
    )
  }

  override def sqr()
  : Unit = foreachChunk(
    LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SQR,
    ArrayEx.sqr(
      values, _, 1,
      _
    )
  )

  override def sqrt()
  : Unit = foreachChunk(
    LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SQRT,
    ArrayEx.sqrt(
      values, _, 1,
      _
    )
  )

  override def stdDev(epsilon: Double)
  : Real = ArrayEx.sampleStdDev(values, epsilon)

  override def sum
  : Real = {
    val result = new AtomicDouble()
    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SUM,
      (off0, length0) => {
        val tmp = ArrayEx.sum(
          values, off0, 1,
          length0
        )
        result.addAndGet(DoubleEx(tmp))
      }
    )
    Real(result.doubleValue)
  }

  @inline
  def foldChunks[T](z0: T, chunkSize: Int)
                   (fnMap: (Int, Int) => T)
                   (fnFold: (T, T) => T)
                   (implicit tagT: ClassTag[T])
  : T = {
    val end0 = values.length
    if (end0 <= chunkSize) {
      val tmp = fnMap(0, end0)
      fnFold(z0, tmp)
    }
    else {
      val tmp = mapChunks(chunkSize, fnMap)
      ArrayEx.foldLeft(
        z0,
        tmp
      )(fnFold)
    }
  }

  @inline
  def foreachChunk(chunkSize: Int,
                   fn:        (Int, Int) => Unit)
  : Unit = {
    val end0 = values.length
    if (end0 <= chunkSize) {
      fn(0, end0)
    }
    else {
      RangeEx.foreachParallel(
        0, end0, chunkSize,
        off0 => fn(off0, Math.min(chunkSize, end0 - off0))
      )
    }
  }

  @inline
  def foreachChunkPair(chunkSize: Int,
                       fn:        (Int, Int, Int) => Unit)
  : Unit = {
    val end0 = values.length
    if (end0 <= chunkSize) {
      fn(0, 0, end0)
    }
    else {
      RangeEx.foreachPairParallel(
        0, end0, chunkSize,
        (i, off0) => fn(i, off0, Math.min(chunkSize, end0 - off0))
      )
    }
  }

  def foreachUnit(fn: (Int, Int, Int) => Unit)
  : Unit = {
    val length0 = layout.noSamples
    val stride0 = layout.size.noValues

    if (stride0 > 1) {
      RangeEx.foreachParallel(
        0, stride0,
        fn(
          _, stride0,
          length0
        )
      )
    }
    else {
      RangeEx.foreach(
        0, stride0,
        fn(
          _, stride0,
          length0
        )
      )
    }
  }

  def foreachUnit(other: RealArrayTensor,
                  fn:    (Int, Int, Int, Int) => Unit)
  : Unit = {
    val length0 = layout.noSamples
    val stride0 = layout.size.noValues
    val length1 = layout.noValues
    val stride1 = layout.size.noChannels

    require(stride0 == stride1)

    if (stride0 > 1) {
      RangeEx.foreachParallel(
        0, stride0,
        fn(
          _, stride0,
          length0,
          length1
        )
      )
    }
    else {
      RangeEx.foreach(
        0, stride0,
        fn(
          _, stride0,
          length0,
          length1
        )
      )
    }
  }

  def foreachChannel(fn: (Int, Int, Int) => Unit)
  : Unit = {
    val length0 = layout.noTuples
    val stride0 = layout.size.noChannels

    if (stride0 > 1) {
      RangeEx.foreachParallel(
        0, stride0,
        fn(
          _, stride0,
          length0
        )
      )
    }
    else {
      RangeEx.foreach(
        0, stride0,
        fn(
          _, stride0,
          length0
        )
      )
    }
  }

  def foreachChannel(other: Tensor,
                     fn:    (Int, Int, Int, Int) => Unit)
  : Unit = {
    val length0 = layout.noTuples
    val stride0 = layout.size.noChannels
    val length1 = layout.noTuples
    val stride1 = layout.size.noChannels

    require(stride0 == stride1)

    if (stride0 > 1) {
      RangeEx.foreachParallel(
        0, stride0,
        fn(
          _, stride0,
          length0,
          length1
        )
      )
    }
    else {
      RangeEx.foreach(
        0, stride0,
        fn(
          _, stride0,
          length0,
          length1
        )
      )
    }
  }

  /*
  def foreachChannelVector(fn: DenseVector[Real] => Unit)
  : Unit = {
    if (layout.size.noChannels > 1) {
      ArrayEx.foreachParallel(
        channelVectors
      )(fn)
    }
    else {
      ArrayEx.foreach(
        channelVectors
      )(fn)
    }
  }

  def foreachChannelVector(other: RealTensor,
                           fn:    (DenseVector[Real], DenseVector[Real]) => Unit)
  : Unit = {
    require(layout.size.noChannels == other.layout.size.noChannels)
    if (layout.size.noChannels > 1) {
      ArrayEx.foreachParallel(
        channelVectors,
        other.channelVectors
      )(fn)
    }
    else {
      ArrayEx.foreach(
        channelVectors,
        other.channelVectors
      )(fn)
    }
  }

  def foreachChannelVectorPair(fn: (Int, DenseVector[Real]) => Unit)
  : Unit = {
    if (layout.size.noChannels > 1) {
      ArrayEx.foreachPairParallel(
        channelVectors
      )(fn)
    }
    else {
      ArrayEx.foreachPair(
        channelVectors
      )(fn)
    }
  }

  def foreachChannelVectorPair(other: RealTensor,
                               fn:    (Int, DenseVector[Real], DenseVector[Real]) => Unit)
    : Unit = {
    require(layout.size.noChannels == other.layout.size.noChannels)
    if (layout.size.noChannels > 1) {
      ArrayEx.foreachPairParallel(
        channelVectors,
        other.channelVectors
      )(fn)
    }
    else {
      ArrayEx.foreachPair(
        channelVectors,
        other.channelVectors
      )(fn)
    }
  }
  */

  @inline
  def foreachSample(fn: (Int, Int) => Unit)
  : Unit = foreachChunk(layout.size.noValues, fn)

  @inline
  def foreachSample(other: Tensor,
                    fn:    (Int, Int, Int, Int) => Unit)
  : Unit = {
    val sampleSize0 = layout.size.noValues
    val offsets0    = Range(0, layout.noValues, sampleSize0)
    val layout1     = other.layout
    val sampleSize1 = layout1.size.noValues
    val offsets1    = Range(0, layout1.noValues, sampleSize1)

    require(layout.noSamples == layout1.noSamples)

    if (layout.noSamples > 1) {
      RangeEx.foreachParallel(
        offsets0,
        offsets1,
        fn(
          _, sampleSize0,
          _, sampleSize1
        )
      )
    }
    else {
      RangeEx.foreach(
        offsets0,
        offsets1,
        fn(
          _, sampleSize0,
          _, sampleSize1
        )
      )
    }
  }

  @inline
  def foreachSample(other:  Tensor,
                    other2: Tensor,
                    fn:     (Int, Int, Int, Int, Int, Int) => Unit)
  : Unit = {
    val sampleSize0 = layout.size.noValues
    val offsets0    = Range(0, layout.noValues, sampleSize0)
    val layout1     = other.layout
    val sampleSize1 = layout1.size.noValues
    val offsets1    = Range(0, layout1.noValues, sampleSize1)
    val layout2     = other2.layout
    val sampleSize2 = layout2.size.noValues
    val offsets2    = Range(0, layout2.noValues, sampleSize2)

    require(
      layout.noSamples == layout1.noSamples &&
      layout.noSamples == layout2.noSamples
    )

    if (layout.noSamples > 1) {
      RangeEx.foreachParallel(
        offsets0,
        offsets1,
        offsets2,
        fn(
          _, sampleSize0,
          _, sampleSize1,
          _, sampleSize2
        )
      )
    }
    else {
      RangeEx.foreach(
        offsets0,
        offsets1,
        offsets2,
        fn(
          _, sampleSize0,
          _, sampleSize1,
          _, sampleSize2
        )
      )
    }
  }

  @inline
  def foreachSamplePair(fn: (Int, Int, Int) => Unit)
  : Unit = foreachChunkPair(layout.size.noValues, fn)

  @inline
  def foreachSamplePair(other: Tensor,
                        fn:    (Int, Int, Int, Int, Int) => Unit)
  : Unit = {
    val sampleSize0 = layout.size.noValues
    val offsets0    = Range(0, layout.noValues, sampleSize0)
    val layout1     = other.layout
    val sampleSize1 = layout1.size.noValues
    val offsets1    = Range(0, layout1.noValues, sampleSize1)

    require(layout.noSamples == layout1.noSamples)

    if (layout.noSamples > 1) {
      RangeEx.foreachPairParallel(
        offsets0,
        offsets1,
        fn(
          _,
          _, sampleSize0,
          _, sampleSize1
        )
      )
    }
    else {
      RangeEx.foreachPair(
        offsets0,
        offsets1,
        fn(
          _,
          _, sampleSize0,
          _, sampleSize1
        )
      )
    }
  }

  @inline
  def foreachTuple(fn: (Int, Int) => Unit)
  : Unit = foreachChunk(layout.size.noChannels, fn)

  /*
  def foreachSampleVector(fn: DenseVector[Real] => Unit)
  : Unit = {
    if (layout.noSamples > 1) {
      ArrayEx.foreachParallel(
        sampleVectors
      )(fn)
    }
    else {
      ArrayEx.foreach(
        sampleVectors
      )(fn)
    }
  }

  def foreachSampleVector(other: RealTensor,
                          fn:    (DenseVector[Real], DenseVector[Real]) => Unit)
  : Unit = {
    require(layout.noSamples == other.layout.noSamples)
    if (layout.noSamples > 1) {
      ArrayEx.foreachParallel(
        sampleVectors,
        other.sampleVectors
      )(fn)
    }
    else {
      MatrixEx.foreachColumnVector(
        valuesMatrix,
        other.valuesMatrix
      )(fn)
    }
  }

  def foreachSampleVectorPair(fn: (Int, DenseVector[Real]) => Unit)
  : Unit = {
    if (layout.noSamples > 1) {
      ArrayEx.foreachPairParallel(
        sampleVectors
      )(fn)
    }
    else {
      ArrayEx.foreachPair(
        sampleVectors
      )(fn)
    }
  }

  def foreachSampleChannelVector(fn: DenseVector[Real] => Unit)
  : Unit = {
    if (layout.noSamples > 1) {
      ArrayEx.foreachParallel(
        sampleChannelVectors
      )(ArrayEx.foreach(_)(fn))
    }
    else {
      ArrayEx.foreach(
        sampleChannelVectors
      )(ArrayEx.foreach(_)(fn))
    }
  }

  def foreachSampleChannelVector(other: RealTensor,
                                 fn:    (DenseVector[Real], DenseVector[Real]) => Unit)
  : Unit = {
    if (layout.noSamples > 1) {
      ArrayEx.foreachParallel(
        sampleChannelVectors,
        other.sampleChannelVectors
      )(ArrayEx.foreach(_, _)(fn))
    }
    else {
      ArrayEx.foreach(
        sampleChannelVectors,
        other.sampleChannelVectors
      )(ArrayEx.foreach(_, _)(fn))
    }
  }

  def foreachSampleChannelVectorPair(fn: (Int, Int, DenseVector[Real]) => Unit)
  : Unit = {
    if (layout.noSamples > 1) {
      ArrayEx.foreachPairParallel(
        sampleChannelVectors
      )((i, arr0) => ArrayEx.foreachPair(arr0)(fn(i, _, _)))
    }
    else {
      ArrayEx.foreachPair(
        sampleChannelVectors
      )((i, arr0) => ArrayEx.foreachPair(arr0)(fn(i, _, _)))
    }
  }

  def foreachSampleChannelVectorPair(other: RealTensor,
                                     fn:    (Int, Int, DenseVector[Real], DenseVector[Real]) => Unit)
  : Unit = {
    if (layout.noSamples > 1) {
      ArrayEx.foreachPairParallel(
        sampleChannelVectors,
        other.sampleChannelVectors
      )((i, arr0, arr1) => ArrayEx.foreachPair(arr0, arr1)(fn(i, _, _, _)))
    }
    else {
      ArrayEx.foreachPair(
        sampleChannelVectors,
        other.sampleChannelVectors
      )((i, arr0, arr1) => ArrayEx.foreachPair(arr0, arr1)(fn(i, _, _, _)))
    }
  }
  */

  @inline
  def mapChunks[T](chunkSize: Int,
                   fn:        (Int, Int) => T)
                  (implicit tagT: ClassTag[T])
  : Array[T] = {
    val end0 = values.length
    if (end0 <= chunkSize) {
      Array(fn(0, end0))
    }
    else {
      RangeEx.mapParallel(
        0, end0, chunkSize
      )(off0 => fn(off0, Math.min(chunkSize, end0 - off0)))
    }
  }

  @inline
  def mapChunkPairs[T](chunkSize: Int,
                       fn:        (Int, Int, Int) => T)
                      (implicit tagT: ClassTag[T])
  : Array[T] = {
    val end0 = values.length
    if (end0 <= chunkSize) {
      Array(fn(0, 0, end0))
    }
    else {
      RangeEx.mapPairsParallel(
        0, end0, chunkSize
      )((i, off0) => fn(i, off0, Math.min(chunkSize, end0 - off0)))
    }
  }

  @inline
  def mapSamples[T](fn: (Int, Int) => T)
                   (implicit tagT: ClassTag[T])
  : Array[T] = mapChunks(layout.size.noValues, fn)

  @inline
  def mapSamples[T](other: Tensor,
                    fn:    (Int, Int, Int, Int) => T)
                   (implicit tagT: ClassTag[T])
  : Array[T] = {
    val result = new Array[T](layout.noSamples)
    foreachSamplePair(
      other,
      (i, off0, length0, off1, length1) => {
        result(i) = fn(off0, length0, off1, length1)
      }
    )
    result
  }

  @inline
  def mapSamplePairs[T](fn: (Int, Int, Int) => T)
                       (implicit tagT: ClassTag[T])
  : Array[T] = mapChunkPairs(layout.size.noValues, fn)

  @inline
  def reduceChunks[T](chunkSize: Int)
                     (fnMap: (Int, Int) => T)
                     (fnReduce: (T, T) => T)
                     (implicit tagT: ClassTag[T])
  : T = {
    val end0 = values.length
    if (end0 <= chunkSize) {
      fnMap(0, end0)
    }
    else {
      val results = mapChunks(chunkSize, fnMap)
      ArrayEx.reduceLeft(
        results
      )(fnReduce)
    }
  }

  @inline
  def tabulate(fn: (Int, Int) => Real)
  : Unit = foreachChunkPair(
    LTU_REAL_TENSOR_CHUNK_SIZE_FOR_TABULATE,
    (i, off0, length0) => {
      ArrayEx.tabulate(
        values, off0, 1,
        length0
      )(fn(i, _))
    }
  )

  @inline
  def transform(fn: Real => Real)
  : Unit = foreachChunk(
    LTU_REAL_TENSOR_CHUNK_SIZE_FOR_TRANSFORM,
    ArrayEx.transform(
      values, _, 1,
      _
    )(fn)
  )

  @inline
  def transform(other: Tensor,
                fn:    (Real, Real) => Real)
  : Unit = {
    require(layout == other.layout)

    val otherValues = other.values

    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_TRANSFORM,
      (off0, length0) => {
        ArrayEx.transform(
          values,      off0, 1,
          otherValues, off0, 1,
          length0
        )(fn)
      }
    )
  }

  @inline
  def transform(other:  Tensor,
                other2: Tensor,
                fn:     (Real, Real, Real) => Real)
  : Unit = {
    require(
      layout == other.layout &&
      layout == other2.layout
    )

    val otherValues1 = other.values
    val otherValues2 = other.values

    foreachChunk(
      LTU_REAL_TENSOR_CHUNK_SIZE_FOR_TRANSFORM,
      (off0, length0) => {
        ArrayEx.transform(
          values,       off0, 1,
          otherValues1, off0, 1,
          otherValues2, off0, 1,
          length0
        )(fn)
      }
    )
  }

  override def ++(other: Tensor)
  : RealArrayTensor = {
    val newLayout = layout ++ other.layout

    val tmp0      = valuesMatrix
    val tmp1      = other.valuesMatrix
    val newValues = MatrixEx.concatRows(tmp0, tmp1)

    RealArrayTensor(newLayout, newValues)
  }

  override def :++(other: Tensor)
  : RealArrayTensor = {
    val newLayout = layout :++ other.layout

    val matrix0   = valuesMatrix
    val matrix1   = other.valuesMatrix
    val tmp0      = MatrixEx.reshape(matrix0, layout.size.noChannels)
    val tmp1      = MatrixEx.reshape(matrix1, other.layout.size.noChannels)
    val newValues = MatrixEx.concatRows(tmp0, tmp1)

    RealArrayTensor(newLayout, newValues)
  }

  override protected def doSlice(tuple0: Int,
                                 result: Tensor)
  : Unit = {
    val unit0 = tuple0 * layout.size.noChannels
    val unit1 = unit0  + result.layout.size.noValues
    val units = unit0 until unit1

    // TODO: Do this without matrix!
    result := valuesMatrix(units, ::)
  }

  override protected def doSliceChannels(channel0: Int,
                                         result:   Tensor)
  : Unit = {
    val channel1 = channel0 + result.layout.size.noChannels
    val channels = channel0 until channel1

    // TODO: Do this without matrix!
    val tmp0 = MatrixEx.reshape(valuesMatrix, layout.size.noChannels)
    result := tmp0(channels, ::)
  }


  // ---------------------------------------------------------------------------
  //    Conversion & extraction methods.
  // ---------------------------------------------------------------------------
  override def toRealArrayTensor
  : RealArrayTensor = copy

  override def asOrToRealArrayTensor
  : RealArrayTensor = this

}

object RealArrayTensor {

  final def apply(values: Array[Real])
  : RealArrayTensor = apply(IndependentTensorLayout.derive(values.length), values)

  final def apply(size: Size, values: Array[Real])
  : RealArrayTensor = apply(size, values.length / size.noValues, values)

  final def apply(size: Size, noSamples: Int, values: Array[Real])
  : RealArrayTensor = apply(IndependentTensorLayout(size, noSamples), values)

  final def apply(layout: IndependentTensorLayout, values: Array[Real])
  : RealArrayTensor = new RealArrayTensor(layout, values)

  final def derive(value0: Real, values: Real*)
  : RealArrayTensor = apply(SeqEx.concat(value0, values))

  final def derive(size: Size, value0: Real, values: Real*)
  : RealArrayTensor = apply(size, SeqEx.concat(value0, values))

  final def derive(size: Size, values: DenseVector[Real])
  : RealArrayTensor = apply(IndependentTensorLayout(size), values.toArray)

  final def derive(size:  Size,
                   part0: DenseVector[Real],
                   parts: DenseVector[Real]*)
  : RealArrayTensor = derive(size, SeqEx.concat(part0, parts))

  final def derive(size: Size, values: DenseMatrix[Real])
  : RealArrayTensor = apply(size, values.cols, values.toArray)

  final def derive(size:  Size,
                   part0: DenseMatrix[Real],
                   parts: DenseMatrix[Real]*)
  : RealArrayTensor = derive(size, SeqEx.concat(part0, parts))

  final def derive(size: Size, parts: Array[Array[Real]])
  : RealArrayTensor = apply(size, parts.length, ArrayEx.concat(parts))

  final def derive(size: Size, parts: Array[DenseVector[Real]])
  : RealArrayTensor = derive(size, VectorEx.multiConcatDenseH(parts))

  final def derive(size: Size, parts: Array[DenseMatrix[Real]])
  : RealArrayTensor = derive(size, MatrixEx.concatColumnsDense(parts))

  final def derive(parts: DenseVector[Real])
  : RealArrayTensor = derive(Size1(1, parts.length), parts)

  final def derive(parts: DenseMatrix[Real])
  : RealArrayTensor = derive(Size1(1, parts.rows), parts)

  final def derive(parts: Array[DenseVector[Real]])
  : RealArrayTensor = derive(VectorEx.multiConcatDenseH(parts))

  final def derive(parts: Array[DenseMatrix[Real]])
  : RealArrayTensor = derive(MatrixEx.concatColumnsDense(parts))

  final def empty
  : RealArrayTensor = zeros(Size1.zero, 0)

  final def fill(size: Size, noSamples: Int, value: Real)
  : RealArrayTensor = fill(IndependentTensorLayout(size, noSamples), value)

  final def fill(layout: IndependentTensorLayout, value: Real)
  : RealArrayTensor = apply(layout, ArrayEx.fill(layout.noValues, value))

  final def fill(size:         Size,
                 noSamples:    Int,
                 distribution: Distribution[Real])
  : RealArrayTensor = fill(
    IndependentTensorLayout(size, noSamples),
    distribution
  )

  final def fill(layout:       IndependentTensorLayout,
                 distribution: Distribution[Real])
  : RealArrayTensor = apply(
    layout,
    ArrayEx.fill(layout.noValues, distribution)
  )

  final def fill(size: Size, noSamples: Int)
                (fn: => Real)
  : RealArrayTensor = fill(
    IndependentTensorLayout(size, noSamples)
  )(fn)

  final def fill(layout: IndependentTensorLayout)
                (fn: => Real)
  : RealArrayTensor = apply(
    layout,
    ArrayEx.fill(
      layout.noValues
    )(fn)
  )

  final def ones(size: Size, noSamples: Int)
  : RealArrayTensor = ones(IndependentTensorLayout(size, noSamples))

  final def ones(layout: IndependentTensorLayout)
  : RealArrayTensor = fill(layout, Real.one)

  final def zeros(size: Size, noSamples: Int)
  : RealArrayTensor = zeros(IndependentTensorLayout(size, noSamples))

  final def zeros(layout: IndependentTensorLayout)
  : RealArrayTensor = apply(layout, new Array[Real](layout.noValues))

  final def zerosLike(tensor: IndependentTensor)
  : RealArrayTensor = zeros(tensor.layout)

}
