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

import org.json4s.JsonAST._
import scala.collection._
import scala.util.hashing._

final class ValueTensorBank(override val segments: SortedMap[Int, ValueTensor])
  extends BankEx[ValueTensorBank, ValueTensor]
    with AutoCloseable
    with CopyableEx[ValueTensorBank] {
  require(!segments.exists(_._2 == null))

  override def toString
  : String = s"ValueTensorBank[${segments.size}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), segments.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ValueTensorBank]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ValueTensorBank =>
      segments == other.segments
    case _ =>
      false
  })

  override def close()
  : Unit = foreachSegment(
    _.close()
  )

  def allocateSibling()
  : ValueTensorBank = allocateSibling(_.createSibling())

  def allocateSibling(allocateFn: ValueTensor => ValueTensor)
  : ValueTensorBank = ValueTensorBank(mapSegments(allocateFn))

  def allocateZeroedSibling()
  : ValueTensorBank = allocateSibling(_.createSiblingAndClear())

  override def copy
  : ValueTensorBank = {
    val result = mapSegments(
      _.copy
    )
    ValueTensorBank(result)
  }

  def noValues
  : Long = foldLeftSegments(
    0L
  )(_ + _.layout.noValues)

  def noValuesPerSegment
  : SortedMap[Int, Int] = mapSegments(
    _.layout.noValues
  )

  def noValuesPerSegmentMax
  : Int = foldLeftSegments(
    0
  )((res, s) => Math.max(res, s.layout.noValues))

  def layout
  : TensorLayoutBank = TensorLayoutBank(layoutPerSegment)

  def layoutPerSegment
  : SortedMap[Int, IndependentTensorLayout] = mapSegments(_.layout)

  def meanEx
  : RealBank = RealBank(meanPerSegment)

  def meanPerSegment
  : SortedMap[Int, Real] = mapSegments(_.mean)

  def approximateMean(rng:          PseudoRNG,
                      noSamplesMax: Int = 1000)
  : Mean = {
    val result = Mean()
    foreachSegment(
      result += _.approximateMean(rng, noSamplesMax)
    )
    result
  }

  def approximateMeanEx(rng:          PseudoRNG,
                        noSamplesMax: Int = 1000)
  : MeanBank = {
    val result = approximateMeanPerSegment(rng, noSamplesMax)
    MeanBank(result)
  }

  def approximateMeanPerSegment(rng:          PseudoRNG,
                                noSamplesMax: Int = 1000)
  : SortedMap[Int, Mean] = mapSegments(
    _.approximateMean(rng, noSamplesMax)
  )

  def approximateMeanAndVariance(rng:          PseudoRNG,
                                 noSamplesMax: Int = 1000)
  : MeanAndVariance = {
    val result = MeanAndVariance()
    foreachSegment(
      result += _.approximateMeanAndVariance(rng, noSamplesMax)
    )
    result
  }

  def approximateMeanAndVarianceEx(rng:          PseudoRNG,
                                   noSamplesMax: Int = 1000)
  : MeanAndVarianceBank = {
    val result = approximateMeanAndVariancePerSegment(rng, noSamplesMax)
    MeanAndVarianceBank(result)
  }
  def approximateMeanAndVariancePerSegment(rng:          PseudoRNG,
                                           noSamplesMax: Int = 1000)
  : SortedMap[Int, MeanAndVariance] = mapSegments(
    _.approximateMeanAndVariance(rng, noSamplesMax)
  )


  // ---------------------------------------------------------------------------
  //    Direct manipulation related.
  // ---------------------------------------------------------------------------
  def indexOf(linearValueNo: Long): (Int, Int) = {
    require(linearValueNo >= 0L)
    var valueNo = linearValueNo
    foreachSegmentPair((i, s) => {
      val noValues = s.layout.noValues
      if (valueNo < noValues) {
        return (i, valueNo.toInt)
      }
      valueNo -= noValues
    })
    throw new IndexOutOfBoundsException
  }

  def apply(segmentNo: Int, valueNo: Int)
  : Real = segments(segmentNo).get(valueNo)

  def apply(valueNo: (Int, Int))
  : Real = apply(valueNo._1, valueNo._2)

  def apply(linearValueNo: Long)
  : Real = apply(indexOf(linearValueNo))

  def update(segmentNo: Int, valueNo: Int, value: Real)
  : Unit = segments(segmentNo).put(valueNo, value)

  def update(valueNo: (Int, Int), value: Real)
  : Unit = update(valueNo._1, valueNo._2, value)

  def update(linearValueNo: Long, value: Real)
  : Unit = update(indexOf(linearValueNo), value)

  def clear()
  : Unit = foreachSegment(
    _.clear()
  )

  def fill(distribution: Distribution[Real])
  : Unit = foreachSegment(
    _.fill(distribution)
  )

  def fill(fn: () => Real, threadSafe: Boolean)
  : Unit = foreachSegment(
    _.fill(fn, threadSafe)
  )

  def integrate(other: ValueTensorBank)
  : Unit = integrate(
    other,
    _ := _
  )

  // TODO: Could be done faster.
  def integrate(other: ValueTensorBank,
                fn:    (ValueTensor, ValueTensor) => Unit)
  : Unit = foreachSegmentEx(
    other
  )(fn, x => {}, y => {})

  /**
    * Shares the memory with this.
    */
  def createView(fn: (Int, ValueTensor) => Boolean)
  : ValueTensorBank = {
    val result = MapEx.filter(
      segments
    )(fn)
    ValueTensorBank(result)
  }

  /*
  def fill(other: ParameterGroup, fn: Real => Real)
  : Unit = foreachSegment(_. other, ArrayEx.fill(_, _)(fn))
  */


  // ---------------------------------------------------------------------------
  //    Operations
  // ---------------------------------------------------------------------------
  def :=(value: Real)
  : Unit = foreachSegment(
    _ := value
  )

  def :=(other: ValueTensorBank)
  : Unit = foreachSegment(
    other
  )(_ := _)

  def +(value: Real)
  : ValueTensorBank = {
    val result = mapSegments(
      _ + value
    )
    ValueTensorBank(result)
  }

  def +(other: ValueTensorBank)
  : ValueTensorBank = {
    val result = zipSegments(
      other
    )(_ + _)
    ValueTensorBank(result)
  }

  def +=(value: Real)
  : Unit = foreachSegment(
    _ += value
  )

  def +=(other: ValueTensorBank)
  : Unit = foreachSegment(
    other
  )(_ += _)


  def add(alpha: Real,
          other: ValueTensorBank)
  : Unit = foreachSegment(
    other
  )(_.add(
    alpha,
    _
  ))

  def add(other: ValueTensorBank, beta: Real)
  : Unit = foreachSegment(
    other
  )(_.add(_, beta))

  def add(alpha: Real,
          other: ValueTensorBank, beta: Real)
  : Unit = foreachSegment(
    other
  )(_.add(alpha, _, beta))

  def add(other0: ValueTensorBank, other1: ValueTensorBank)
  : Unit = foreachSegment(
    other0,
    other1
  )(_.add(_, _))

  def add(alpha:  Real,
          other0: ValueTensorBank, other1: ValueTensorBank)
  : Unit = foreachSegment(
    other0,
    other1
  )(_.add(alpha, _, _))

  def add(other0: ValueTensorBank, other1: ValueTensorBank, beta: Real)
  : Unit = foreachSegment(
    other0,
    other1
  )(_.add(_, _, beta))


  def add(alpha:  Real,
          other0: ValueTensorBank, other1: ValueTensorBank, beta: Real)
  : Unit = foreachSegment(
    other0,
    other1
  )(_.add(alpha, _, _, beta))

  def unary_-()
  : ValueTensorBank = {
    val result = mapSegments(
      -_
    )
    ValueTensorBank(result)
  }

  def -(other: ValueTensorBank)
  : ValueTensorBank = {
    val result = zipSegments(
      other
    )(_ - _)
    ValueTensorBank(result)
  }

  def -=(other: ValueTensorBank)
  : Unit = foreachSegment(
    other
  )(_ -= _)

  def *(value: Real)
  : ValueTensorBank = {
    val result = mapSegments(
      _ * value
    )
    ValueTensorBank(result)
  }

  def :*(other: ValueTensorBank)
  : ValueTensorBank = {
    val result = zipSegments(
      other
    )(_ :* _)
    ValueTensorBank(result)
  }

  def *=(value: Real)
  : Unit = foreachSegment(
    _ *= value
  )

  def :*=(other: ValueTensorBank)
  : Unit = foreachSegment(
    other
  )(_ :*= _)

  def multiply(other: ValueTensorBank, beta: Real)
  : Unit = foreachSegment(
    other
  )(_.multiply(_, beta))

  def :/(other: ValueTensorBank)
  : ValueTensorBank = {
    val result = zipSegments(
      other
    )(_ :/ _)
    ValueTensorBank(result)
  }

  def :/=(other: ValueTensorBank)
  : Unit = foreachSegment(
    other
  )(_ :/= _)

  def reciprocal()
  : ValueTensorBank = {
    val result = mapSegments(_.reciprocal())
    ValueTensorBank(result)
  }

  def subtractR(value: Real)
  : Unit = foreachSegment(
    _.subtractR(value)
  )

  def divide(epsilon0: Real,
             other: ValueTensorBank)
  : Unit = foreachSegment(
    other
  )(_.divide(epsilon0, _))

  def divide(other: ValueTensorBank, epsilon1: Real)
  : Unit = foreachSegment(
    other
  )(_.divide(_, epsilon1))

  def divide(epsilon0: Real,
             other: ValueTensorBank, epsilon1: Real)
  : Unit = foreachSegment(
    other
  )(_.divide(epsilon0, _, epsilon1))

  def divideR(value: Real)
  : Unit = foreachSegment(
    _.divideR(value)
  )

  def lerp(other: ValueTensorBank, t: Real)
  : Unit = foreachSegment(
    other
  )(_.lerp(_, t))

  def lerp(other0: ValueTensorBank, other1: ValueTensorBank, t: Real)
  : Unit = foreachSegment(
    other0, other1
  )(_.lerp(_, _, t))

  def dot(other: ValueTensorBank)
  : Real = foldLeftSegments(
    Real.zero,
    other
  )((res, a, b) => res + a.dot(b))

  def l1Norm(epsilon: Double)
  : Real = foldLeftSegments(
    Real.zero
  )(_ + _.l1Norm(epsilon))

  def l2Norm(epsilon: Double)
  : Real = Real(Math.sqrt(l2NormSq + epsilon))

  def l2NormSq
  : Real = foldLeftSegments(
    Real.zero
  )(_ + _.l2NormSq)

  def sum
  : Real = foldLeftSegments(
    Real.zero
  )(_ + _.sum)

  def min()
  : Real = mapReduceLeftSegments(
    _.min()
  )(Math.min)

  def min(other: ValueTensorBank)
  : Unit = foreachSegment(
    other
  )(_.min(_))

  def max()
  : Real = mapReduceLeftSegments(
    _.max()
  )(Math.max)

  def max(other: ValueTensorBank)
  : Unit = foreachSegment(
    other
  )(_.max(_))

  def maxByAbs(other: ValueTensorBank)
  : Unit = foreachSegment(
    other
  )(_.maxByAbs(_))

  def abs()
  : Unit = foreachSegment(
    _.abs()
  )

  def sqr()
  : Unit = foreachSegment(
    _.sqr()
  )

  def sqrt()
  : Unit = foreachSegment(
    _.sqrt()
  )

  /*
  @inline
  def foldLeftSegmentsParallel[T](z0: T)
                                 (fn0: (T, DenseVector[Real]) => T, fn1: (T, T) => T)
  : T = parallelSegments.aggregate(z0)((z0, pair) => fn0(z0, pair._2), fn1)

  def foldLeftValues[T](z0: T)(fn: (T, Real) => T)
  : T = foldLeftSegments(z0)(ArrayEx.foldLeft(_, _)(fn))

  def foldLeftValues[T](z0: T, other: ParameterGroup)(fn: (T, Real, Real) => T)
  : T = foldLeftSegments(z0, other)(ArrayEx.foldLeft(_, _, _)(fn))

  def foldLeftValuePairs[T](z0: T)(fn: (T, Int, Int, Real) => T)
  : T = foldLeftSegmentPairs(z0)(
    (z0, i, a) => ArrayEx.foldLeftPairs(z0, a)(fn(_, i, _, _))
  )

  def foldLeftValuePairs[T](z0: T, other: ParameterGroup)
                           (fn: (T, Int, Int, Real, Real) => T)
  : T = foldLeftSegmentPairs(z0, other)(
    (z0, i, a, b) => ArrayEx.foldLeftPairs(z0, a, b)(fn(_, i, _, _, _))
  )

  def foldLeftLinearValuePairs[T](z0: T)(fn: (T, Long, Real) => T): T = {
    var z = z0
    foreachLinearValuePair(
      (i, v) => z = fn(z, i, v)
    )
    z
  }
  */

  /*
  def foreachSegmentEx(n0:    Long,
                       other: ParameterBank,
                       n1:    Long,
                       fn:    (ValueTensor, Long, ValueTensor, Long) => Unit)
  : Unit = foreachSegment(
    other
  )(fn(_, n0, _, n1))
  */

  /*
  @deprecated
  def foreachValue(fn: Real => Unit)
  : Unit = foreachSegment(
    s => ArrayEx.foreach(s.values)(fn)
  )

  @deprecated
  def foreachValuePair(fn: (Int, Int, Real) => Unit)
  : Unit = foreachSegmentPair(
    (i, s) => ArrayEx.foreachPair(s.values)(fn(i, _, _))
  )

  @deprecated
  def foreachValuePair(fn: (Long, Real) => Unit): Unit = {
    var valueNo = 0L
    foreachValue(v => {
      fn(valueNo, v)
      valueNo += 1L
    })
  }
  */

  /*
  def mapValues(fn: Real => Real)
  : ParameterGroup = mapSegments(s => {
    val values = ArrayEx.map(s.values)(fn)
    ValueTensor(s.size, s.noSamples, values)
  })

  def mapValuePairs(fn: (Int, Int, Real) => Real)
  : ParameterGroup = mapSegmentPairs((i, s) => {
    val values = ArrayEx.mapPairs(s.values)(fn(i, _, _))
    ValueTensor(s.size, s.noSamples, values)
  })
  */

  /*
  @inline
  def foreachSegmentParallel(fn: DenseVector[Real] => Unit)
  : Unit = parallelSegments.foreach(s => fn(s._2))

  @inline
  def foreachSegmentParallel(other: ParameterGroup,
                             fn:    (DenseVector[Real], DenseVector[Real]) => Unit)
  : Unit = parallelSegments.zip(other.parallelSegments).foreach(pair => {
    val a = pair._1
    val b = pair._2
    require(a._1 == b._1 && a._2.offset == b._2.offset)
    fn(a._2, b._2)
  })

  def foreachSegmentParallel(other:  ParameterGroup,
                             other2: ParameterGroup,
                             fn:     (DenseVector[Real], DenseVector[Real], DenseVector[Real]) => Unit)
  : Unit = parallelSegments.zip(
    other.parallelSegments.zip(other2.parallelSegments)
  ).foreach(pair => {
    val a = pair._1
    val b = pair._2._1
    val c = pair._2._2
    require(a._1 == b._1 && a._2.offset == b._2.offset)
    require(a._1 == c._1 && a._2.offset == c._2.offset)
    fn(a._2, b._2, c._2)
  })
  */

  /*
  @inline
  def reduceLeftValues(fn: (Real, Real) => Real)
  : Real = {
    val tmp = MapEx.mapValues(segments)(
      s => ArrayEx.reduceLeft(s.values)(fn)
    )
    MapEx.reduceLeftValues(tmp)(fn)
  }
  */

  def tabulate(fn: (Int, Int) => Real)
  : Unit = foreachSegmentPair(
    (i, s) => {
      val n = s.layout.noValues
      var j = 0
      while (j < n) {
        s.put(j, fn(i, j))
        j += 1
      }
    }
  )
  /*
  def tabulateLinear(fn: Long => Real): Unit = {
    var index = 0L
    foreachSegment(ArrayEx.fill(_)({
      val tmp = fn(index)
      index += 1L
      tmp
    }))
  }
  */

  /*
  def transformValues(fn: Real => Real)
  : Unit = foreachSegment(_.transform(fn))
  */

  /*
  @inline
  def transformValues(other: ParameterGroup, fn: (Real, Real) => Real)
  : Unit = foreachSegment(other, ArrayEx.transform(_, _)(fn))

  @inline
  def transformValues(other:  ParameterGroup,
                      other2: ParameterGroup,
                      fn:     (Real, Real, Real) => Real)
  : Unit = foreachSegment(other, other2, ArrayEx.transform(_, _, _)(fn))

  @inline
  def transformValuesParallel(other: ParameterGroup,
                              fn:    (Real, Real) => Real)
  : Unit = foreachSegmentParallel(other, VectorEx.transform(_, _)(fn))

  @inline
  def transformValuesParallel(other:  ParameterGroup,
                              other2: ParameterGroup,
                              fn:     (Real, Real, Real) => Real)
  : Unit = foreachSegmentParallel(
    other, other2, VectorEx.transform(_, _, _)(fn)
  )

  @inline
  def transformValuePairs(fn: (Int, Int, Real) => Real)
  : Unit = foreachSegmentPair(
    (i, s) => ArrayEx.transformPairs(s)(fn(i, _, _))
  )

  @inline
  def transformLinearValuePairs(fn: (Long, Real) => Real): Unit = {
    var index = 0L
    foreachSegment(_.transform(v => {
      val tmp = fn(index, v)
      index += 1L
      tmp
    }))
  }
  */


  /*
  @inline
  def zipValues(other: ParameterGroup, fn: (Real, Real) => Real)
  : ParameterGroup = zipSegments(other, (a, b) => {
    require(a.size == b.size)
    val values = ArrayEx.zip(a.values, b.values)(fn)
    ValueTensor(a.size, a.noSamples, values)
  })

  @inline
  def zipValuePairs(other: ParameterGroup,
                    fn:    (Int, Int, Real, Real) => Real)
  : ParameterGroup = zipSegmentPairs(other, (i, a, b) => {
    require(a.size == b.size)
    val values = ArrayEx.zipPairs(a.values, b.values)(fn(i, _, _, _))
    ValueTensor(a.size, a.noSamples, values)
  })
  */


  // ---------------------------------------------------------------------------
  //     Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, ValueTensor])
  : ValueTensorBank = ValueTensorBank(banks)

  override protected def doToJson(segment: ValueTensor)
  : JValue = segment.toJson

  def toMap
  : SortedMap[(Int, Int), Real] = {
    val builder = SortedMap.newBuilder[(Int, Int), Real]
    foreachSegmentPair((i, s) => {
      ArrayEx.foreachPair(
        s.values
      )((j, v) => builder += Tuple2((i, j), v))
    })
    builder.result()
  }

  def toLinearMap
  : SortedMap[Long, Real] = {
    val builder = SortedMap.newBuilder[Long, Real]
    var i = 0L
    foreachSegment(
      s => ArrayEx.foreach(s.values)(v => {
        builder += Tuple2(i, v)
        i       += 1
      })
    )
    builder.result()
  }

  def asOrToRealTensorBankJVM
  : ValueTensorBank = {
    val result = mapSegments(_.asOrToRealArrayTensor)
    ValueTensorBank(result)
  }

  def toRealTensorBankJVM
  : ValueTensorBank = {
    val result = mapSegments(_.toRealArrayTensor)
    ValueTensorBank(result)
  }

  /*
  def makePortable
  : Portable[ParameterBank] = {
    var readOnly = false
    val result = mapSegments(s => {
      val p = s.makePortable
      readOnly ||= p.isReadOnly
      p.unwrap(false)
    })
    Portable(ParameterBank(result), readOnly)
  }
  */

  /*
  def dump(stream: ObjectOutputStream)
  : Unit = {
    stream.writeInt(segments.size)
    foreachSegmentPair((i, s) => {
      stream.writeInt(i)
      stream.writeObject(s.layout)
      stream.writeObject(s.values)
    })
  }
  */

  /*
  def toSparseMatrix: CSCMatrix[Real] = {
    val noValues = {
      val tmp = this.noValues
      require(tmp <= Int.MaxValue)
      tmp.toInt
    }

    val rows   = MapEx.foldLeftValues(0, segments)(
      (a, b) => Math.max(a, b.length)
    )
    val cols   = segments.lastKey + 1
    val result = CSCMatrix.zeros[Real](rows, cols)
    result.reserve(noValues)
    foreachValuePair(
      (segNo, valNo, value) => result.update(valNo, segNo, value)
    )
    result
  }
  */

}

object ValueTensorBank
  extends BankExCompanion[ValueTensorBank, ValueTensor]
    with JsonSerializableCompanionEx[ValueTensorBank] {

  final override def apply(segments: SortedMap[Int, ValueTensor])
  : ValueTensorBank = new ValueTensorBank(segments)

  final override protected def doDerive(json: JValue)
  : ValueTensor = RealTensor.derive(json)

  /*
  final def zeros(layout: ParameterGroupLayout)
  : ParameterGroup = ParameterGroup(
    MapEx.mapValues(layout.segments)(Array.ofDim[Real])
  )

  final def ones(layout: ParameterGroupLayout)
  : ParameterGroup = ParameterGroup(MapEx.mapValues(layout.segments)(
    noValues => ArrayEx.fill(noValues, Real.one)
  ))

  final def zerosLike(reference: ParameterGroup, x: Int): ParameterGroup = apply(
    reference.segments.mapValues(s => DVec.zeros(s.length))
  )

  final def onesLike(reference: ParameterGroup): ParameterGroup = apply(
    reference.segments.mapValues(s => DVec.ones(s.length))
  )

  // TODO: Remove explicit implementations of operators and put them here as universal functions!

  implicit val canLerp
  = new OpLerp.Impl3[ParameterGroup, ParameterGroup, Real, ParameterGroup] {

    override def apply(a: ParameterGroup, b: ParameterGroup, t: Real)
    : ParameterGroup = a.zipSegments(b,
      lerp(_, _, t)
    )

  }

  implicit val canLerpInto
  = new OpLerp.InPlaceImpl3[ParameterGroup, ParameterGroup, Real] {

    // TODO: Benchmark this. Pure Scala implementation could be faster!
    override def apply(a: ParameterGroup, b: ParameterGroup, t: Real)
    : Unit = a.foreachSegment(b,
      lerp.inPlace(_, _, t)
    )
    //a.data(i) *= Real.one - t
    //axpy(t, b.buffers(i), a.buffers(i))

  }

  implicit val canScaleAdd
  = new scaleAdd.Impl3[ParameterGroup, Real, ParameterGroup, ParameterGroup] {

    override def apply(y: ParameterGroup, a: Real, x: ParameterGroup)
    : ParameterGroup = y.zipSegments(x, (y, x) => {
      val tmp = y.copy
      axpy(a, x, tmp)
      tmp
    })

  }

  implicit val canScaleAddInto
  = new scaleAdd.InPlaceImpl3[ParameterGroup, Real, ParameterGroup] {

    override def apply(y: ParameterGroup, a: Real, x: ParameterGroup)
    : Unit = x.foreachSegment(y,
      axpy(a, _, _)
    )

  }

  implicit def canTraverseValues
  : CanTraverseValues[ParameterGroup, Real]
  = new CanTraverseValues[ParameterGroup, Real] {

    override def isTraversableAgain(from: ParameterGroup): Boolean = true

    override def traverse(from: ParameterGroup, fn: ValuesVisitor[Real])
    : Unit = from.foreachParameter(fn.visit)

  }

  implicit def canTraverseKeyValuePairs
  : CanTraverseKeyValuePairs[ParameterGroup, (Int, Int), Real]
  = new CanTraverseKeyValuePairs[ParameterGroup, (Int, Int), Real] {

    override def isTraversableAgain(from: ParameterGroup): Boolean = true

    override def traverse(from: ParameterGroup,
                          fn:   KeyValuePairsVisitor[(Int, Int), Real])
    : Unit = from.foreachParameterPair(
      (s, i, v) => fn.visit((s, i), v)
    )

  }

  implicit def canMapValues(implicit cmv: CanMapValues[DVec, Real, Real, DVec])
  : CanMapValues[ParameterGroup, Real, Real, ParameterGroup]
  = new CanMapValues[ParameterGroup, Real, Real, ParameterGroup] {

    override def map(from: ParameterGroup, fn: Real => Real)
    : ParameterGroup = from.mapParameters(fn)

    override def mapActive(from: ParameterGroup, fn: Real => Real)
    : ParameterGroup = from.mapParameters(fn)

  }
  */

  /*
  final def restore(stream: ObjectInputStream)
  : RealTensorBank = {
    val builder    = SortedMap.newBuilder[Int, RealArrayTensor]
    val noSegments = stream.readInt()
    var i = 0
    while (i < noSegments) {
      val segNo  = stream.readInt()
      val layout = stream.readObject().asInstanceOf[IndependentTensorLayout]
      val values = stream.readObject().asInstanceOf[Array[Real]]
      builder += Tuple2(segNo, RealArrayTensor(layout, values))
      i += 1
    }
    apply(builder.result())
  }
  */

}

final class ValueTensorBankBuilder
  extends BankExBuilder[ValueTensorBank, ValueTensor] {

  override def result()
  : ValueTensorBank = ValueTensorBank(toSortedMap)

}

object ValueTensorBankBuilder {

  final def apply()
  : ValueTensorBankBuilder = new ValueTensorBankBuilder

}
