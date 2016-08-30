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

import breeze.linalg.DenseMatrix
import org.json4s.JsonAST._
import scala.collection._
import scala.reflect._
import scala.util.hashing._

final class TensorTable(private val _tensors: Array[Tensor])
  extends TensorEx[TensorTable]
    with DependentTensor
    with TableLike[Tensor]
    with Serializable {
  require(!ArrayEx.contains(_tensors, null))

  override def repr
  : TensorTable = this

  override def dependsOn(other: Tensor)
  : Boolean = super.dependsOn(other) || ArrayEx.exists(
    _tensors
  )(_.dependsOn(other))

  override def toString
  : String = s"TensorTable[${_tensors.length}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), ArrayEx.hashCode(_tensors))

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: TensorTable =>
      ArrayEx.compare(
        _tensors,
        other._tensors
      )
    case _ =>
      false
  })

  override protected def doClose()
  : Unit = {
    foreachTensor(_.close())
    super.doClose()
  }

  override def createSibling(newLayout: TensorLayout)
  : TensorTable = throw new NotImplementedError

  override def createSiblingAndClear(newLayout: TensorLayout)
  : TensorTable = throw new NotImplementedError

  override def copy
  : TensorTable = TensorTable(mapTensors(_.copy))


  // ---------------------------------------------------------------------------
  //    Data access related.
  // ---------------------------------------------------------------------------
  override def length
  : Int = _tensors.length

  override def getEntry(index: Int)
  : Tensor = {
    val arrayIndex = {
      if (index >= 0) {
        index
      }
      else {
        _tensors.length + index
      }
    }
    require(arrayIndex >= 0 && arrayIndex < _tensors.length)
    _tensors(arrayIndex)
  }

  override def iterator
  : Iterator[Tensor] =  _tensors.iterator

  override def platform
  : PlatformTable = PlatformTable(mapTensors(_.platform))

  override def layout
  : TensorLayoutTable = TensorLayoutTable(mapTensors(_.layout))

  override def values
  : Array[Real] = {
    val noValues = foldLeftTensors(0)(_ + _.layout.noValues)
    val result = new Array[Real](noValues)
    get(result)
    result
  }

  override def valuesMatrix
  : DenseMatrix[Real] = {
    val result = values
    new DenseMatrix(result.length, 1, result)
  }

  override def valuesMatrixEx
  : DenseMatrix[Real] = {
    val result = values
    new DenseMatrix(result.length, 1, result)
  }

  override def get(valueNo: Int)
  : Real = {
    require(valueNo >= 0)
    var n = valueNo
    var i = 0
    while (i < _tensors.length) {
      val tensor = _tensors(i)
      val size   = tensor.layout.noValues
      if (n < size) {
        return tensor.get(n)
      }
      n -= size
      i += 1
    }
    throw new IndexOutOfBoundsException
  }

  override def get(result: Array[Real], offset: Int, stride: Int)
  : Unit = {
    var off = offset
    foreachTensor(tensor => {
      tensor.get(result, off, stride)
      off += tensor.layout.noValues * stride
    })
  }

  override def put(valueNo: Int, value: Real)
  : Unit = {
    require(valueNo >= 0)
    var n = valueNo
    var i = 0
    while (i < _tensors.length) {
      val ten  = _tensors(i)
      val size = ten.layout.noValues
      if (n < size) {
        ten.put(n, value)
        return
      }
      n -= size
      i += 1
    }
    throw new IndexOutOfBoundsException
  }

  override def put(array: Array[Real], offset: Int, stride: Int)
  : Unit = {
    foldLeftTensors(offset)((offset, tensor) => {
      tensor.put(array, offset, stride)
      offset + tensor.layout.noValues * stride
    })
  }

  override def clear()
  : Unit = {
    foreachTensor(
      _.clear()
    )
  }

  override def fill(fn: () => Real, threadSafe: Boolean)
  : Unit = {
    foreachTensor(
      _.fill(fn, threadSafe)
    )
  }


  // ---------------------------------------------------------------------------
  //    Basic operations.
  // ---------------------------------------------------------------------------
  override def reshape(newSize: Size)
  : TensorTable = throw new NotImplementedError

  override def apply(index: Int)
  : TensorTable = TensorTable.derive(_tensors(index).copy)

  override def apply(indices: Range)
  : TensorTable = {
    val result = ArrayEx.map(
      ArrayEx.slice(_tensors, indices)
    )(_.copy)
    TensorTable(result)
  }

  override def splitSamples
  : Array[Tensor] = throw new NotImplementedError

  override def concat(other: Tensor)
  : TensorTable = {
    val result = zipTensors(
      other
    )(_.concat(_))
    TensorTable(result)
  }

  override def concat[T <: Tensor](others: Array[T])
  : TensorTable = {
    def getTensor(tensor: Tensor, index: Int)
    : Tensor = tensor match {
      case tensor: TensorTable =>
        tensor._tensors(index)
      case _ =>
        throw new MatchError(tensor)
    }
    val newTensors = ArrayEx.mapPairs(_tensors)(
      (i, tensor) => tensor.concat(ArrayEx.map(others)(getTensor(_, i)))
    )
    TensorTable(newTensors)
  }

  override protected def doSet(value: Real)
  : Unit = foreachTensor(
    _ := value
  )

  override protected def doSet(other: Tensor)
  : Unit = {
    foreachTensor(
      other
    )(_ := _)
  }

  override protected def doSet(other: Tensor, beta: Real)
  : Unit = foreachTensor(
    other
  )(_.set(_, beta))

  override protected def doSet(other0: Tensor, other1: Tensor)
  : Unit = foreachTensor(
    other0, other1
  )(_.set(_, _))

  override protected def doSet(other0: Tensor, other1: Tensor, beta: Real)
  : Unit = foreachTensor(
    other0, other1
  )(_.set(_, _, beta))

  override protected def doAdd(value: Real)
  : Unit = foreachTensor(
    _ += value
  )

  override protected def doAdd(other: Tensor)
  : Unit = foreachTensor(
    other
  )(_ += _)

  override protected def doAdd(alpha: Real,
                               other: Tensor)
  : Unit = foreachTensor(
    other
  )(_.add(alpha, _))

  override protected def doAdd(other: Tensor, beta: Real)
  : Unit = foreachTensor(
    other
  )(_.add(_, beta))

  override protected def doAdd(alpha: Real,
                               other: Tensor, beta: Real)
  : Unit = foreachTensor(
    other
  )(_.add(alpha, _, beta))

  override protected def doAdd(other0: Tensor, other1: Tensor)
  : Unit = foreachTensor(
    other0,
    other1
  )(_.add(_, _))

  override protected def doAdd(alpha:  Real,
                               other0: Tensor, other1: Tensor)
  : Unit = foreachTensor(
    other0,
    other1
  )(_.add(alpha, _, _))

  override protected def doAdd(other0: Tensor, other1: Tensor, beta: Real)
  : Unit = foreachTensor(
    other0,
    other1
  )(_.add(_, _, beta))

  override protected def doAdd(alpha:  Real,
                               other0: Tensor, other1: Tensor, beta: Real)
  : Unit = foreachTensor(
    other0,
    other1
  )(_.add(alpha, _, _, beta))

  override protected def doSubtract(other: Tensor)
  : Unit = foreachTensor(
    other
  )(_ -= _)

  override def subtractR(value: Real)
  : Unit = foreachTensor(
    _.subtractR(value)
  )

  override protected def doMultiply(value: Real)
  : Unit = foreachTensor(
    _ *= value
  )

  override protected def doMultiply(other: Tensor)
  : Unit = foreachTensor(
    other
  )(_ :*= _)

  override protected def doMultiply(other: Tensor, beta: Real)
  : Unit = foreachTensor(
    other
  )(_.multiply(_, beta))

  override protected def doDivide(other: Tensor)
  : Unit = foreachTensor(
    other
  )(_ :/= _)

  override protected def doDivide(epsilon0: Real,
                                  other:    Tensor)
  : Unit = foreachTensor(
    other
  )(_.divide(epsilon0, _))

  override protected def doDivide(other: Tensor, epsilon1: Real)
  : Unit = foreachTensor(
    other
  )(_.divide(_, epsilon1))

  override protected def doDivide(epsilon0: Real,
                                  other:    Tensor, epsilon1: Real)
  : Unit = foreachTensor(
    other
  )(_.divide(epsilon0, _, epsilon1))

  override protected def doDivideR(value: Real)
  : Unit = foreachTensor(
    _.divideR(value)
  )

  override protected def doDot(other: Tensor)
  : Real = foldLeftTensors(
    Real.zero,
    other
  )(_ + _.dot(_))

  override protected def doLerp(other: Tensor, t: Real)
  : Unit = foreachTensor(
    other
  )(_.lerp(_, t))

  override protected def doLerp(other0: Tensor, other1: Tensor, t: Real)
  : Unit = foreachTensor(
    other0,
    other1
  )(_.lerp(_, _, t))


  // ---------------------------------------------------------------------------
  //    Fancy operations.
  // ---------------------------------------------------------------------------
  override def abs()
  : Unit = foreachTensor(_.abs())

  override def approximateMean(rng: PseudoRNG, noSamplesMax: Int)
  : Mean = {
    mapReduceTensors(
      _.approximateMean(rng, noSamplesMax)
    )((a, b) => { a += b; a})
  }

  override def approximateMeanAndVariance(rng: PseudoRNG, noSamplesMax: Int)
  : MeanAndVariance = {
    mapReduceTensors(
      _.approximateMeanAndVariance(rng, noSamplesMax)
    )((a, b) => { a += b; a})
  }

  override def l1Norm(epsilon: Double)
  : Real = foldLeftTensors(
    Real.zero
  )(_ + _.l1Norm(epsilon))

  override def l2Norm(epsilon: Double)
  : Real = Real(Math.sqrt(l2NormSq + epsilon))

  override def l2NormSq
  : Real = foldLeftTensors(
    Real.zero
  )(_ + _.l2NormSq)

  override def max()
  : Real = mapReduceTensors(
    _.max()
  )(Math.max)

  override def max(other: Tensor)
  : Unit = foreachTensor(
    other
  )(_.max(_))

  override def maxAbs()
  : Real = mapReduceTensors(
    _.maxAbs()
  )(Math.max)

  override protected def doMaxByAbs(other: Tensor)
  : Unit = foreachTensor(
    other
  )(_.maxByAbs(_))

  override def mean
  : Real = throw new NotImplementedError

  override def min()
  : Real = mapReduceTensors(
    _.min()
  )(Math.min)

  override def min(other: Tensor)
  : Unit = foreachTensor(
    other
  )(_.min(_))

  override def sign()
  : Unit = foreachTensor(_.sign())

  override def sqr()
  : Unit = foreachTensor(_.sqr())

  override def sqrt()
  : Unit = foreachTensor(_.sqrt())

  override def sum
  : Real = foldLeftTensors(
    Real.zero
  )(_ + _.sum)

  override def stdDev(epsilon: Double)
  : Real = throw new NotImplementedError

  @inline
  def foldLeftTensors[T](z0: T)
                        (fn: (T, Tensor) => T)
  : T = ArrayEx.foldLeft(
    z0,
    _tensors
  )(fn)

  @inline
  def foldLeftTensors[T](z0: T, other: Tensor)
                        (fn: (T, Tensor, Tensor) => T)
  : T = other match {
    case other: TensorTable =>
      foldLeftTensors(
        z0,
        other
      )(fn)
    case _ =>
      throw new MatchError(other)
  }

  @inline
  def foldLeftTensors[T](z0: T, other: TensorTable)
                        (fn: (T, Tensor, Tensor) => T)
  : T = ArrayEx.foldLeft(
    z0,
    _tensors,
    other._tensors
  )(fn)

  @inline
  def foreachTensor(fn: Tensor => Unit)
  : Unit = ArrayEx.foreach(
    _tensors
  )(fn)

  @inline
  def foreachTensor(other: Tensor)
                   (fn:    (Tensor, Tensor) => Unit)
  : Unit = other match {
    case other: TensorTable =>
      foreachTensor(
        other
      )(fn)
    case _ =>
      throw new MatchError(other)
  }

  @inline
  def foreachTensor(other: TensorTable)
                   (fn:    (Tensor, Tensor) => Unit)
  : Unit = ArrayEx.foreach(
    _tensors,
    other._tensors
  )(fn)

  @inline
  def foreachTensor(other:  Tensor,
                    other2: Tensor)
                   (fn:     (Tensor, Tensor, Tensor) => Unit)
  : Unit = (other, other2) match {
    case (other: TensorTable, other2: TensorTable) =>
      foreachTensor(
        other,
        other2
      )(fn)
    case _ =>
      throw new MatchError((other, other2))
  }

  @inline
  def foreachTensor(other:  TensorTable,
                    other2: TensorTable)
                   (fn:     (Tensor, Tensor, Tensor) => Unit)
  : Unit = ArrayEx.foreach(
    _tensors,
    other._tensors,
    other2._tensors
  )(fn)

  @inline
  def foreachTensorPair(fn: (Int, Tensor) => Unit)
  : Unit = ArrayEx.foreachPair(
    _tensors
  )(fn)

  @inline
  def foreachTensorPair(other: TensorTable,
                        fn:    (Int, Tensor, Tensor) => Unit)
  : Unit = ArrayEx.foreachPair(
    _tensors,
    other._tensors
  )(fn)

  @inline
  def mapTensors[T](fn: Tensor => T)
                   (implicit tagT: ClassTag[T])
  : Array[T] = ArrayEx.map(
    _tensors
  )(fn)

  @inline
  def mapReduceTensors[T](fnMap: Tensor => T)
                         (fnReduce: (T, T) => T)
                         (implicit tagT: ClassTag[T])
  : T = ArrayEx.mapReduce(
    _tensors
  )(fnMap)(fnReduce)

  @inline
  def zipTensors[T](other: Tensor)
                   (fn: (Tensor, Tensor) => T)
                   (implicit tagT: ClassTag[T])
  : Array[T] = other match {
    case other: TensorTable =>
      zipTensors(
        other
      )(fn)
    case _ =>
      throw new MatchError(other)
  }

  @inline
  def zipTensors[T](other: TensorTable)
                   (fn: (Tensor, Tensor) => T)
                   (implicit tagT: ClassTag[T])
  : Array[T] = ArrayEx.zip(
    _tensors,
    other._tensors
  )(fn)

  @inline
  override def ++(other: Tensor)
  : TensorTable = other match {
    case other: TensorTable =>
      ++(other)
    case _ =>
      throw new MatchError(other)
  }

  @inline
  def ++(other: TensorTable)
  : TensorTable = {
    val result = zipTensors(
      other
    )(_ ++ _)
    TensorTable(result)
  }

  @inline
  override def :++(other: Tensor)
  : TensorTable = other match {
    case other: TensorTable =>
      :++(other)
    case _ =>
      throw new MatchError(other)
  }

  @inline
  def :++(other: TensorTable)
  : TensorTable = {
    val result = zipTensors(
      other
    )(_ :++ _)
    TensorTable(result)
  }

  override protected def doSlice(tuple0: Int,
                                 result: Tensor)
  : Unit = result match {
    case result: TensorTable =>
      foreachTensor(
        result
      )(_.slice(tuple0, _))
    case _ =>
      throw new MatchError(result)
  }

  override protected def doSliceChannels(channel0: Int,
                                         result:   Tensor)
  : Unit = result match {
    case result: TensorTable =>
      foreachTensor(
        result
      )(_.sliceChannels(channel0, _))
    case _ =>
      throw new MatchError(result)
  }


  // ---------------------------------------------------------------------------
  //    Conversion & extraction methods.
  // ---------------------------------------------------------------------------
  override protected def doToJson()
  : List[JField] = List(
    Json.field("tensors", _tensors)
  )

  override def toValueTensor
  : ValueTensor = RealArrayTensor(layout.makeIndependent, values)

  override def asOrToValueTensor
  : ValueTensor = RealArrayTensor(layout.makeIndependent, values)

  override def toRealArrayTensor
  : RealArrayTensor = RealArrayTensor(layout.makeIndependent, values)

  override def asOrToRealArrayTensor
  : RealArrayTensor = RealArrayTensor(layout.makeIndependent, values)

}

object TensorTable {

  final def apply(tensors: Array[Tensor])
  : TensorTable = new TensorTable(tensors)

  final def derive(tensor0: Tensor)
  : TensorTable = apply(Array(tensor0))

  final def derive(tensor0: Tensor, tensor1: Tensor)
  : TensorTable = apply(Array(tensor0, tensor1))

}
