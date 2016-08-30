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

import breeze.linalg.{CSCMatrix, DenseMatrix, SparseVector}
import edu.latrobe.sizes._
import edu.latrobe.time._

/**
 * Encapsulates sparse mini-batch related results without meta data.
  *
  * @param valuesMatrixEx Internal representation of the values.
 */
final class SparseRealMatrixTensor(val          size:           Size,
                                   override val valuesMatrixEx: CSCMatrix[Real])
  extends TensorEx[SparseRealMatrixTensor]
    with RealTensor
    with Serializable {
  require(valuesMatrixEx.rows == size.noValues)

  override def repr
  : SparseRealMatrixTensor = this

  override def toString
  : String = s"SparseRealTensor[$size  x ${valuesMatrixEx.cols}]"

  override def platform
  : JVM.type = JVM

  override def copy
  : SparseRealMatrixTensor = SparseRealMatrixTensor(size, valuesMatrixEx.copy)

  override def createSibling(newLayout: TensorLayout)
  : SparseRealMatrixTensor = SparseRealMatrixTensor.zeros(newLayout.makeIndependent)

  override def createSiblingAndClear(newLayout: TensorLayout)
  : SparseRealMatrixTensor = SparseRealMatrixTensor.zeros(newLayout.makeIndependent)

  override def values
  : Array[Real] = MatrixEx.toArray(valuesMatrixEx)

  override def valuesMatrix
  : DenseMatrix[Real] = valuesMatrixEx.toDense

  override def get(valueNo: Int)
  : Real = valuesMatrixEx(
    valueNo / size.noValues,
    valueNo % size.noValues
  )

  override def get(result: Array[Real], offset: Int, stride: Int)
  : Unit = {
    ArrayEx.fill(
      result, offset, stride,
      Real.zero,
      layout.noValues
    )
    MatrixEx.copyActive(
      result, offset, stride,
      valuesMatrixEx
    )
  }

  override def put(valueNo: Int, value: Real)
  : Unit = valuesMatrixEx.update(
    valueNo / size.noValues,
    valueNo % size.noValues,
    value
  )

  override def put(array: Array[Real], offset: Int, stride: Int)
  : Unit = throw new UnsupportedOperationException

  @transient
  override lazy val layout
  : IndependentTensorLayout = IndependentTensorLayout(size, valuesMatrixEx.cols)

  override def clear()
  : Unit = valuesMatrixEx := Real.zero

  override def fill(fn: () => Real, threadSafe: Boolean)
  : Unit =  throw new UnsupportedOperationException


  // ---------------------------------------------------------------------------
  //    Basic operations.
  // ---------------------------------------------------------------------------
  override def reshape(newSize: Size)
  : SparseRealMatrixTensor = SparseRealMatrixTensor(newSize, valuesMatrixEx)

  override def apply(index: Int)
  : SparseRealMatrixTensor = SparseRealMatrixTensor.derive(
    size,
    MatrixEx.extractColumnVector(valuesMatrixEx, index)
  )

  override def apply(indices: Range)
  : SparseRealMatrixTensor = SparseRealMatrixTensor.derive(
    size,
    MatrixEx.extractColumnVectors(valuesMatrixEx, indices)
  )

  override def splitSamples
  : Array[Tensor] = ArrayEx.map(
    MatrixEx.extractColumnVectors(valuesMatrixEx)
  )(SparseRealMatrixTensor.derive(size, _))

  override def doSet(value: Real)
  : Unit = valuesMatrixEx := value

  override protected def doSet(other: Tensor)
  : Unit = other match {
    case other: SparseRealMatrixTensor =>
      valuesMatrixEx := other.valuesMatrixEx
    case _ =>
      throw new UnsupportedOperationException
  }

  override protected def doSet(other: Tensor, beta: Real)
  : Unit = {
    doSet(other)
    doMultiply(beta)
  }

  override protected def doSet(other0: Tensor, other1: Tensor)
  : Unit = {
    doSet(other0)
    doMultiply(other1)
  }

  override protected def doSet(other0: Tensor, other1: Tensor, beta: Real)
  : Unit = {
    doSet(other0, other1)
    doMultiply(beta)
  }

  override protected def doAdd(value: Real)
  : Unit = valuesMatrixEx += value

  override protected def doAdd(other: Tensor)
  : Unit = other match {
    case other: SparseRealMatrixTensor =>
      valuesMatrixEx += other.valuesMatrixEx
    case _ =>
      throw new UnsupportedOperationException
  }

  override protected def doAdd(alpha: Real, other: Tensor)
  : Unit = {
    doMultiply(alpha)
    doAdd(other)
  }

  override protected def doAdd(other: Tensor, beta: Real)
  : Unit = other match {
    case other: SparseRealMatrixTensor =>
      valuesMatrixEx += (other.valuesMatrixEx * beta)
    case _ =>
      throw new UnsupportedOperationException
  }

  override protected def doAdd(alpha: Real,
                               other: Tensor, beta: Real)
  : Unit = {
    doMultiply(alpha)
    doAdd(other, beta)
  }

  override protected def doAdd(other0: Tensor, other1: Tensor)
  : Unit = other0 match {
    case other0: SparseRealMatrixTensor =>
      other1 match {
        case other1: SparseRealMatrixTensor =>
          valuesMatrixEx :+= (other0.valuesMatrixEx :* other1.valuesMatrixEx)
        case _ =>
          throw new UnsupportedOperationException
      }
    case _ =>
      throw new UnsupportedOperationException
  }

  override protected def doAdd(alpha:  Real,
                               other0: Tensor, other1: Tensor)
  : Unit = {
    doMultiply(alpha)
    doAdd(other0, other1)
  }

  override protected def doAdd(other0: Tensor, other1: Tensor, beta: Real)
  : Unit = other0 match {
    case other0: SparseRealMatrixTensor =>
      other1 match {
        case other1: SparseRealMatrixTensor =>
          val tmp = other0.valuesMatrixEx :* other1.valuesMatrixEx
          tmp *= beta
          valuesMatrixEx += tmp
        case _ =>
          throw new UnsupportedOperationException
      }
    case _ =>
      throw new UnsupportedOperationException
  }

  override protected def doAdd(alpha: Real,
                               other0: Tensor, other1: Tensor, beta: Real)
  : Unit = {
    doMultiply(alpha)
    doAdd(other0, other1, beta)
  }

  override protected def doSubtract(other: Tensor)
  : Unit = other match {
    case other: SparseRealMatrixTensor =>
      valuesMatrixEx -= other.valuesMatrixEx
    case _ =>
      throw new UnsupportedOperationException
  }

  override def subtractR(value: Real)
  : Unit = {
    MatrixEx.subtract(
      value,
      valuesMatrixEx
    )
  }

  override protected def doMultiply(value: Real)
  : Unit = valuesMatrixEx *= value

  override protected def doMultiply(other: Tensor)
  : Unit = other match {
    case other: SparseRealMatrixTensor =>
      valuesMatrixEx :*= other.valuesMatrixEx
    case _ =>
      throw new UnsupportedOperationException
  }

  override protected def doMultiply(other: Tensor, beta: Real)
  : Unit = other match {
    case other: SparseRealMatrixTensor =>
      valuesMatrixEx :*= other.valuesMatrixEx
      valuesMatrixEx *= beta
    case _ =>
      throw new UnsupportedOperationException
  }

  override protected def doDivide(other: Tensor)
  : Unit = other match {
    case other: SparseRealMatrixTensor =>
      valuesMatrixEx :/= other.valuesMatrixEx
    case _ =>
      throw new UnsupportedOperationException
  }

  override protected def doDivide(epsilon0: Real,
                                  other: Tensor)
  : Unit = other match {
    case other: SparseRealMatrixTensor =>
      valuesMatrixEx += epsilon0
      valuesMatrixEx :/= other.valuesMatrixEx
    case _ =>
      throw new UnsupportedOperationException
  }

  override protected def doDivide(other: Tensor, epsilon1: Real)
  : Unit = other match {
    case other: SparseRealMatrixTensor =>
      valuesMatrixEx :/= (other.valuesMatrixEx + epsilon1)
    case _ =>
      throw new UnsupportedOperationException
  }

  override protected def doDivide(epsilon0: Real,
                                  other: Tensor, epsilon1: Real)
  : Unit = other match {
    case other: SparseRealMatrixTensor =>
      valuesMatrixEx += epsilon0
      valuesMatrixEx :/= (other.valuesMatrixEx + epsilon1)
    case _ =>
      throw new UnsupportedOperationException
  }

  override protected def doDivideR(value: Real)
  : Unit = {
    MatrixEx.divide(
      value,
      valuesMatrixEx
    )
  }

  override protected def doDot(other: Tensor)
  : Real = throw new NotImplementedError

  override protected def doLerp(other: Tensor, t: Real)
  : Unit = doAdd(
    Real.one - t,
    other, t
  )

  override protected def doLerp(other0: Tensor, other1: Tensor, t: Real)
  : Unit = doAdd(
    Real.one - t,
    other0, other1, t
  )


  // ---------------------------------------------------------------------------
  //    Fancy operations.
  // ---------------------------------------------------------------------------
  override def abs()
  : Unit = MatrixEx.abs(valuesMatrixEx)

  override def l1Norm(epsilon: Double)
  : Real = throw new NotImplementedError

  override def l2Norm(epsilon: Double)
  : Real = throw new NotImplementedError

  override def l2NormSq
  : Real = throw new NotImplementedError

  override def max()
  : Real = MatrixEx.max(valuesMatrixEx)

  override def max(other: Tensor)
  : Unit = throw new NotImplementedError

  override def maxAbs()
  : Real = MatrixEx.maxAbs(valuesMatrixEx)

  override protected def doMaxByAbs(other: Tensor)
  : Unit = throw new NotImplementedError

  override def mean
  : Real = breeze.stats.mean(valuesMatrixEx)

  override def min()
  : Real = MatrixEx.min(valuesMatrixEx)

  override def min(other: Tensor)
  : Unit = throw new NotImplementedError

  override def sign()
  : Unit = throw new UnsupportedOperationException

  override def sqr()
  : Unit = MatrixEx.sqr(valuesMatrixEx)

  override def sqrt()
  : Unit = MatrixEx.sqrt(valuesMatrixEx)

  override def stdDev(epsilon: Double)
  : Real = ArrayEx.sampleStdDev(values, epsilon)

  override def sum
  : Real = MatrixEx.sum(valuesMatrixEx)

  override def ++(other: Tensor)
  : SparseRealMatrixTensor = throw new NotImplementedError

  override def :++(other: Tensor)
  : SparseRealMatrixTensor = throw new NotImplementedError

  override protected def doSlice(tuple0: Int,
                                 result: Tensor)
  : Unit = throw new NotImplementedError

  override protected def doSliceChannels(channel0: Int,
                                         result:   Tensor)
  : Unit = throw new NotImplementedError

  override def concat(other: Tensor): SparseRealMatrixTensor = other match {
    case other: SparseRealMatrixTensor =>
      SparseRealMatrixTensor(
        size, MatrixEx.concatColumns(valuesMatrixEx, other.valuesMatrixEx)
      )
    case _ =>
      throw new MatchError(other)
  }

  override def concat[T <: Tensor](others: Array[T])
  : SparseRealMatrixTensor = {
    val otherMatrices = ArrayEx.map(others)({
      case other: SparseRealMatrixTensor =>
        require(other.layout.size == size)
        other.valuesMatrixEx
      case _ =>
        throw new MatchError(others)
    })

    val newValues = SeqEx.concat(valuesMatrixEx, otherMatrices)
    SparseRealMatrixTensor.derive(size, newValues)
  }

  override def toRealArrayTensor
  : RealArrayTensor = RealArrayTensor(layout, values)

  override def asOrToRealArrayTensor
  : RealArrayTensor = RealArrayTensor(layout, values)

  /*
  override def makePortable
  : Portable[RealArrayTensor] = Portable.writable(toRealArrayTensor)
  */

}

object SparseRealMatrixTensor {

  final def apply(valuesMatrixEx: CSCMatrix[Real])
  : SparseRealMatrixTensor = apply(Size1(1, valuesMatrixEx.rows), valuesMatrixEx)

  final def apply(size: Size, valuesMatrixEx: CSCMatrix[Real])
  : SparseRealMatrixTensor = new SparseRealMatrixTensor(size, valuesMatrixEx)

  final def derive(size: Size, parts: SparseVector[Real])
  : SparseRealMatrixTensor = apply(size, VectorEx.toMatrix(parts))

  final def derive(size:  Size,
                   part0: SparseVector[Real],
                   parts: SparseVector[Real]*)
  : SparseRealMatrixTensor = derive(size, SeqEx.concat(part0, parts))

  final def derive(size: Size, parts: Array[SparseVector[Real]])
  : SparseRealMatrixTensor = apply(size, VectorEx.multiConcatSparseH(parts))

  /*
  final def derive(size: Size, parts: Seq[SparseVector[Real]])
  : SparseRealTensor = apply(size, VectorEx.multiConcatSparseH(parts))
  */

  final def derive(size: Size, parts: Array[CSCMatrix[Real]])
  : SparseRealMatrixTensor = apply(size, MatrixEx.concatColumnsSparse(parts))

  /*
  final def derive(size: Size, parts: Seq[CSCMatrix[Real]])
  : SparseRealTensor = apply(size, MatrixEx.multiConcatSparseH(parts))
  */

  final def derive(parts: SparseVector[Real])
  : SparseRealMatrixTensor = derive(Size1(1, parts.length), parts)

  final def derive(part0: SparseVector[Real],
                   parts: SparseVector[Real]*)
  : SparseRealMatrixTensor = derive(SeqEx.concat(part0, parts))

  final def derive(parts: Array[SparseVector[Real]])
  : SparseRealMatrixTensor = apply(VectorEx.multiConcatSparseH(parts))

  /*
  final def derive(parts: Seq[SparseVector[Real]])
  : SparseRealTensor = apply(VectorEx.multiConcatSparseH(parts))
  */

  final def derive(part0: CSCMatrix[Real],
                   parts: CSCMatrix[Real]*)
  : SparseRealMatrixTensor = derive(SeqEx.concat(part0, parts))

  final def derive(parts: Array[CSCMatrix[Real]])
  : SparseRealMatrixTensor = apply(MatrixEx.concatColumnsSparse(parts))

  /*
  final def derive(parts: Seq[CSCMatrix[Real]])
  : SparseRealTensor = apply(MatrixEx.multiConcatSparseH(parts))
  */

  final def empty
  : SparseRealMatrixTensor = zeros(Size1.zero, 0)

  final def zeros(size: Size, noSamples: Int)
  : SparseRealMatrixTensor = apply(
    size, CSCMatrix.zeros[Real](size.noValues, noSamples)
  )

  final def zeros(layout: IndependentTensorLayout)
  : SparseRealMatrixTensor = zeros(layout.size, layout.noSamples)

}