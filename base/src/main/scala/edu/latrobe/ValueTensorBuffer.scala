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

final class ValueTensorBuffer(override val banks: SortedMap[Int, ValueTensorBank])
  extends BufferEx[ValueTensorBuffer, ValueTensorBank, ValueTensor]
    with AutoCloseable
    with CopyableEx[ValueTensorBuffer]
    with JsonSerializable {
  require(!banks.exists(_._2 == null))

  override def toString
  : String = s"ValueTensorBuffer[${banks.size}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), banks.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ValueTensorBuffer]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ValueTensorBuffer =>
      banks == other.banks
    case _ =>
      false
  })

  override def close()
  : Unit = foreachBank(
    _.close()
  )

  def allocateSibling()
  : ValueTensorBuffer = allocateSibling(
    _.createSibling()
  )

  def allocateSibling(allocateFn: ValueTensor => ValueTensor)
  : ValueTensorBuffer = {
    val result = mapBanks(
      _.allocateSibling(allocateFn)
    )
    ValueTensorBuffer(result)
  }

  def allocateZeroedSibling()
  : ValueTensorBuffer = allocateSibling(
    _.createSiblingAndClear()
  )

  override def copy
  : ValueTensorBuffer = {
    val result = mapBanks(
      _.copy
    )
    ValueTensorBuffer(result)
  }

  def noValues
  : Long = foldLeftBanks(
    0L
  )(_ + _.noValues)

  def noValuesPerBank
  : SortedMap[Int, Long] = mapBanks(_.noValues)

  def noValuesPerSegmentMax
  : Int = foldLeftBanks(
    0
  )((res, b) => Math.max(res, b.noValuesPerSegmentMax))

  def layout
  : TensorLayoutBuffer = TensorLayoutBuffer(layoutPerBank)

  def layoutPerBank
  : SortedMap[Int, TensorLayoutBank] = mapBanks(_.layout)

  def meanEx
  : RealBuffer = RealBuffer(mapBanks(_.meanEx))

  def meanExPerBank
  : SortedMap[Int, RealBank] = mapBanks(_.meanEx)

  def approximateMean(rng:          PseudoRNG,
                      noSamplesMax: Int = 1000)
  : Mean = {
    val result = Mean()
    foreachBank(
      result += _.approximateMean(rng, noSamplesMax)
    )
    result
  }

  def approximateMeanPerBank(rng:          PseudoRNG,
                             noSamplesMax: Int = 1000)
  : SortedMap[Int, Mean] = mapBanks(_.approximateMean(rng, noSamplesMax))

  def approximateMeanEx(rng:          PseudoRNG,
                        noSamplesMax: Int = 1000)
  : MeanBuffer = MeanBuffer(approximateMeanExPerBank(rng, noSamplesMax))

  def approximateMeanExPerBank(rng:          PseudoRNG,
                               noSamplesMax: Int = 1000)
  : SortedMap[Int, MeanBank] = mapBanks(_.approximateMeanEx(rng, noSamplesMax))

  def approximateMeanAndVariance(rng:          PseudoRNG,
                                 noSamplesMax: Int = 1000)
  : MeanAndVariance = {
    val result = MeanAndVariance()
    foreachBank(
      result += _.approximateMeanAndVariance(rng, noSamplesMax)
    )
    result
  }

  def approximateMeanAndVariancePerBank(rng:          PseudoRNG,
                                        noSamplesMax: Int = 1000)
  : SortedMap[Int, MeanAndVariance] = mapBanks(
    _.approximateMeanAndVariance(rng, noSamplesMax)
  )

  def approximateMeanAndVarianceEx(rng:          PseudoRNG,
                                   noSamplesMax: Int = 1000)
  : MeanAndVarianceBuffer = {
    val result = approximateMeanAndVarianceExPerBank(rng, noSamplesMax)
    MeanAndVarianceBuffer(result)
  }

  def approximateMeanAndVarianceExPerBank(rng:          PseudoRNG,
                                          noSamplesMax: Int = 1000)
  : SortedMap[Int, MeanAndVarianceBank] = mapBanks(
    _.approximateMeanAndVarianceEx(rng, noSamplesMax)
  )

  /*
  def segments: SortedMap[(Int, Int), ValueTensor] = {
    val builder = SortedMap.newBuilder[(Int, Int), ValueTensor]
    foreachGroupPair((i, g) => {
      val segments = g.segments.map(
        kv=> (i, kv._1) -> kv._2
      )
      builder ++= segments
    })
    builder.result()
  }
  */


  // ---------------------------------------------------------------------------
  //    Direct manipulation related.
  // ---------------------------------------------------------------------------
  def indexOf(linearValueNo: Long): (Int, Int, Int) = {
    require(linearValueNo >= 0L)
    var valueNo = linearValueNo
    foreachBankPair((i, g) => {
      if (valueNo < g.noValues) {
        val index = g.indexOf(valueNo)
        return (i, index._1, index._2)
      }
      valueNo -= g.noValues
    })
    throw new IndexOutOfBoundsException
  }

  def apply(bankNo: Int, segmentNo: Int, valueNo: Int)
  : Real = banks(bankNo)(segmentNo, valueNo)

  def apply(valueNo: (Int, Int, Int))
  : Real = apply(valueNo._1, valueNo._2, valueNo._3)

  def apply(linearValueNo: Long)
  : Real = apply(indexOf(linearValueNo))

  def update(bankNo: Int, segmentNo: Int, valueNo: Int, value: Real)
  : Unit = banks(bankNo).update(segmentNo, valueNo, value)

  def update(valueNo: (Int, Int, Int), value: Real)
  : Unit = update(valueNo._1, valueNo._2, valueNo._3, value)

  def update(linearValueNo: Long, value: Real)
  : Unit = update(indexOf(linearValueNo), value)

  def clear()
  : Unit = foreachBank(_.clear())

  def fill(distribution: Distribution[Real])
  : Unit = foreachBank(
    _.fill(distribution)
  )
  def fill(fn: () => Real, threadSafe: Boolean)
  : Unit = foreachBank(
    _.fill(fn, threadSafe)
  )

  def integrate(values: ValueTensorBuffer)
  : Unit = integrate(values, _.integrate(_))

  // TODO: Could be done faster.
  def integrate(other: ValueTensorBuffer,
                fn:    (ValueTensorBank, ValueTensorBank) => Unit)
  : Unit = foreachBankEx(
    other
  )(fn, x => {}, y => {})

  def integrateSegments(other: ValueTensorBuffer,
                        fn:    (ValueTensor, ValueTensor) => Unit)
  : Unit = integrate(
    other,
    _.integrate(_, fn)
  )

  /**
    * Shares the memory with this.
    */
  def createView(fn: (Int, Int, ValueTensor) => Boolean)
  : ValueTensorBuffer = {
    val result = mapBankPairs(
      (i, b) => b.createView(fn(i, _, _))
    )
    ValueTensorBuffer(result)
  }

  /*
  def fill(other: ParameterBuffer, fn: Real => Real)
  : Unit = foreachGroup(other, _.fill(_, fn))
  */

  @inline
  def tabulate(fn: (Int, Int, Int) => Real)
  : Unit = foreachBankPair(
    (i, g) => g.tabulate(fn(i, _, _))
  )

  /*
  @inline
  def tabulateLinear(fn: (Int, Long) => Real)
  : Unit = foreachGroupPair(
    (i, g) => g.tabulateLinear(fn(i, _))
  )
  */

  // ---------------------------------------------------------------------------
  //    Operations
  // ---------------------------------------------------------------------------
  def :=(value: Real)
  : Unit = foreachBank(
    _ := value
  )

  def :=(other: ValueTensorBuffer)
  : Unit = foreachBank(
    other
  )(_ := _)

  def +(value: Real)
  : ValueTensorBuffer = {
    val result = mapBanks(
      _ + value
    )
    ValueTensorBuffer(result)
  }

  def +(other: ValueTensorBuffer)
  : ValueTensorBuffer = {
    val result = zipBanks(
      other
    )(_ + _)
    ValueTensorBuffer(result)
  }

  def +=(value: Real)
  : Unit = foreachBank(
    _ += value
  )

  def +=(other: ValueTensorBuffer)
  : Unit = foreachBank(
    other
  )(_ += _)

  def add(alpha: Real,
          other: ValueTensorBuffer)
  : Unit = foreachBank(
    other
  )(_.add(alpha, _))

  def add(other: ValueTensorBuffer, beta: Real)
  : Unit = foreachBank(
    other
  )(_.add(_, beta))

  def add(alpha: Real,
          other: ValueTensorBuffer, beta: Real)
  : Unit = foreachBank(
    other
  )(_.add(alpha, _, beta))

  def add(other0: ValueTensorBuffer, other1: ValueTensorBuffer)
  : Unit = foreachBank(
    other0,
    other1
  )(_.add(_, _))

  def add(alpha: Real,
          other0: ValueTensorBuffer, other1: ValueTensorBuffer)
  : Unit = foreachBank(
    other0,
    other1
  )(_.add(alpha, _, _))

  def add(alpha: Real,
          other0: ValueTensorBuffer, other1: ValueTensorBuffer, beta: Real)
  : Unit = foreachBank(
    other0,
    other1
  )(_.add(alpha, _, _, beta))


  def add(other0: ValueTensorBuffer, other1: ValueTensorBuffer, beta: Real)
  : Unit = foreachBank(
    other0,
    other1
  )(_.add(_, _, beta))

  def unary_-()
  : ValueTensorBuffer = {
    val result = mapBanks(
      -_
    )
    ValueTensorBuffer(result)
  }

  def -(other: ValueTensorBuffer)
  : ValueTensorBuffer = {
    val result = zipBanks(
      other
    )(_ - _)
    ValueTensorBuffer(result)
  }

  def -=(other: ValueTensorBuffer)
  : Unit = foreachBank(
    other
  )(_ -= _)

  def *(factor: Real)
  : ValueTensorBuffer = {
    val result = mapBanks(
      _ * factor
    )
    ValueTensorBuffer(result)
  }

  def :*(other: ValueTensorBuffer)
  : ValueTensorBuffer = {
    val result = zipBanks(
      other
    )(_ :* _)
    ValueTensorBuffer(result)
  }

  def *=(factor: Real)
  : Unit = foreachBank(
    _ *= factor
  )

  def :*=(other: ValueTensorBuffer)
  : Unit = foreachBank(
    other
  )(_ :*= _)

  def multiply(other: ValueTensorBuffer, beta: Real)
  : Unit = foreachBank(
    other
  )(_.multiply(_, beta))

  def :/(other: ValueTensorBuffer)
  : ValueTensorBuffer = {
    val result = zipBanks(
      other
    )(_ :/ _)
    ValueTensorBuffer(result)
  }

  def :/=(other: ValueTensorBuffer)
  : Unit = foreachBank(
    other
  )(_ :/= _)

  def reciprocal(value: Real)
  : ValueTensorBuffer = {
    val result = mapBanks(
      _.reciprocal()
    )
    ValueTensorBuffer(result)
  }

  def subtractR(value: Real)
  : Unit = foreachBank(
    _.subtractR(value)
  )

  def divide(epsilon0: Real,
             other: ValueTensorBuffer, epsilon1: Real)
  : Unit = foreachBank(
    other
  )(_.divide(epsilon0, _, epsilon1))

  def divide(other: ValueTensorBuffer, epsilon1: Real)
  : Unit = foreachBank(
    other
  )(_.divide(_, epsilon1))

  def divideR(value: Real)
  : Unit = foreachBank(
    _.divideR(value)
  )

  def lerp(other: ValueTensorBuffer, t: Real)
  : Unit = foreachBank(
    other
  )(_.lerp(_, t))

  def lerp(other0: ValueTensorBuffer, other1: ValueTensorBuffer, t: Real)
  : Unit = foreachBank(
    other0,
    other1
  )(_.lerp(_, _, t))

  def dot(other: ValueTensorBuffer)
  : Real = foldLeftBanks(
    Real.zero,
    other
  )((res, a, b) => res + a.dot(b))

  def l1Norm(epsilon: Double)
  : Real = foldLeftBanks(
    Real.zero
  )(_ + _.l1Norm(epsilon))

  def l2Norm(epsilon: Double)
  : Real = Real(Math.sqrt(l2NormSq + epsilon))

  def l2NormSq
  : Real = foldLeftBanks(
    Real.zero
  )(_ + _.l2NormSq)

  def sum
  : Real = foldLeftBanks(
    Real.zero
  )(_ + _.sum)

  def min()
  : Real = mapReduceLeftBanks(_.min())(Math.min)

  def min(other: ValueTensorBuffer)
  : Unit = foreachBank(
    other
  )(_.min(_))

  def max()
  : Real = mapReduceLeftBanks(_.max())(Math.max)

  def max(other: ValueTensorBuffer)
  : Unit = foreachBank(
    other
  )(_.max(_))

  def maxByAbs(other: ValueTensorBuffer)
  : Unit = foreachBank(
    other
  )(_.maxByAbs(_))

  /*
  def foldLeftValues[T](z0: T)(fn: (T, Real) => T): T = foldLeftGroups(z0)(
    (z0, a) => a.foldLeftValues(z0)(fn)
  )

  def foldLeftValues[T](z0: T, other: ParameterBuffer)
                       (fn: (T, Real, Real) => T)
  : T = foldLeftGroups(z0, other)(
    (z0, a, b) => a.foldLeftValues(z0, b)(fn)
  )

  def foldLeftValuePairs[T](z0: T)(fn: (T, Int, Int, Int, Real) => T)
  : T = foldLeftGroupPairs(z0)((z0, i, a) => a.foldLeftValuePairs(z0)(
    fn(_, i, _, _, _)
  ))

  def foldLeftLinearValuePairs[T](z: T)(fn: (T, Long, Real) => T): T = {
    var result = z
    foreachLinearValuePair(
      (i, v) => result = fn(result, i, v)
    )
    result
  }
  */


  /*
  @inline
  def foldLeftSegmentsParallel[T](z0: T)
                                 (fn0: (T, DenseVector[Real]) => T,
                                  fn1: (T, T) => T)
  : T = foldLeftGroups(z0)(
    (z0, a) => a.foldLeftSegmentsParallel(z0)(fn0, fn1)
  )
  */

  /*
  @inline
  def foreachSegmentParallel(fn: DenseVector[Real] => Unit)
  : Unit = foreachGroup( _.foreachSegmentParallel(fn))

  @inline
  def foreachSegmentParallel(other: ParameterBuffer,
                             fn:    (DenseVector[Real], DenseVector[Real]) => Unit)
  : Unit = foreachGroup(other, _.foreachSegmentParallel(_, fn))
  */

  /*
  @inline
  def foreachSegmentParallel(other:  ParameterBuffer,
                             other2: ParameterBuffer,
                             fn:     (DenseVector[Real], DenseVector[Real], DenseVector[Real]) => Unit)
  : Unit = foreachGroup(other, other2, _.foreachSegmentParallel(_, _, fn))
  */

  /*
  def foreachSegmentEx(n0:    Long,
                       other: ParameterBuffer,
                       n1:    Long,
                       fn:    (ValueTensor, Long, ValueTensor, Long) => Unit)
  : Unit = foreachSegment(other, fn(_, n0, _, n1))
  */

  /*
  @deprecated
  def foreachValue(fn: Real => Unit)
  : Unit = foreachGroup(_.foreachValue(fn))

  @deprecated
  def foreachValuePair(fn: (Int, Int, Int, Real) => Unit)
  : Unit = foreachGroupPair(
    (i, g) => g.foreachValuePair(fn(i, _, _, _))
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
  @inline
  def transformValues(fn: Real => Real)
  : Unit = foreachGroup(_.transformValues(fn))

  @inline
  def transformValues(other: ParameterBuffer, fn: (Real, Real) => Real)
  : Unit = foreachGroup(other, _.transformValues(_, fn))

  @inline
  def transformValues(other:  ParameterBuffer,
                      other2: ParameterBuffer,
                      fn:     (Real, Real, Real) => Real)
  : Unit = foreachGroup(other, other2, _.transformValues(_, _, fn))

  @inline
  def transformValuesParallel(other: ParameterBuffer,
                              fn:    (Real, Real) => Real)
  : Unit = foreachGroup(other, _.transformValuesParallel(_, fn))

  @inline
  def transformValuesParallel(other:  ParameterBuffer,
                              other2: ParameterBuffer,
                              fn:     (Real, Real, Real) => Real)
  : Unit = foreachGroup(other, other2, _.transformValuesParallel(_, _, fn))

  @inline
  def transformValuePairs(fn: (Int, Int, Int, Real) => Real)
  : Unit = foreachGroupPair(
    (i, g) => g.transformValuePairs(fn(i, _, _, _))
  )

  def transformLinearValuePairs(fn: (Long, Real) => Real): Unit = {
    var valueNo = 0L
    foreachSegment(_.transform(v => {
      val tmp = fn(valueNo, v)
      valueNo += 1L
      tmp
    }))
  }
  */


  /*
  def zipSegments(other: ParameterBuffer,
                  fn:    (Array[Real], Array[Real]) => Array[Real])
  : ParameterBuffer = zipGroups(other, _.zipSegments(_, fn))

  def zipSegmentPairs(other: ParameterBuffer,
                      fn:    (Int, Int, Array[Real], Array[Real]) => Array[Real])
  : ParameterBuffer = zipGroupPairs(other,
    (i, a, b) => a.zipSegmentPairs(b, fn(i, _, _, _)
  ))

  def zipValues(other: ParameterBuffer, fn: (Real, Real) => Real)
  : ParameterBuffer = zipGroups(other, _.zipValues(_, fn))

  def zipValuePairs(other: ParameterBuffer,
                    fn:    (Int, Int, Int, Real, Real) => Real)
  : ParameterBuffer = zipGroupPairs(other,
    (i, a, b) => a.zipValuePairs(b, fn(i, _, _, _, _))
  )
  */


  // ---------------------------------------------------------------------------
  //     Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, ValueTensorBank])
  : ValueTensorBuffer = ValueTensorBuffer(banks)

  def toPairMap
  : SortedMap[(Int, Int, Int), Real] = {
    val builder = SortedMap.newBuilder[(Int, Int, Int), Real]
    foreachSegmentPair((i, j, s) => {
      ArrayEx.foreachPair(
        s.values
      )((k, v) => builder += Tuple2((i, j, k), v))
    })
    builder.result()
  }

  def toLinearPairMap
  : SortedMap[Long, Real] = {
    val builder = SortedMap.newBuilder[Long, Real]
    var i = 0L
    foreachSegment(s => {
      ArrayEx.foreach(
        s.values
      )(v => {
        builder += Tuple2(i, v)
        i       += 1L
      })
    })
    builder.result()
  }

  /*
  def ensureSerializable()
  : ParameterBuffer = {
    val result = mapBanks(_.ensureSerializable())
    ParameterBuffer(result)
  }
  */

  def asOrToRealTensorBufferJVM
  : ValueTensorBuffer = {
    val result = mapBanks(
      _.asOrToRealTensorBankJVM
    )
    ValueTensorBuffer(result)
  }

  def toRealTensorBufferJVM
  : ValueTensorBuffer = {
    val result = mapBanks(
      _.toRealTensorBankJVM
    )
    ValueTensorBuffer(result)
  }

  /*
  def makePortable
  : Portable[ParameterBuffer] = {
    var readOnly = false
    val result = mapBanks(s => {
      val p = s.makePortable
      readOnly ||= p.isReadOnly
      p.unwrap(false)
    })
    Portable(ParameterBuffer(result), readOnly)
  }
  */

  /*
  def dump(stream: ObjectOutputStream)
  : Unit = {
    stream.writeInt(banks.size)
    foreachBankPair((i, b) => {
      stream.writeInt(i)
      b.dump(stream)
    })
  }
  */

}

object ValueTensorBuffer {

  final def apply(banks: SortedMap[Int, ValueTensorBank])
  : ValueTensorBuffer = new ValueTensorBuffer(banks)

  final def derive(bank0: (Int, ValueTensorBank))
  : ValueTensorBuffer = apply(SortedMap(bank0))

  final def derive(bank0: (Int, ValueTensorBank), banks: (Int, ValueTensorBank)*)
  : ValueTensorBuffer = {
    val builder = SortedMap.newBuilder[Int, ValueTensorBank]
    builder += bank0
    builder ++= banks
    apply(builder.result())
  }

  final def derive(json: JValue)
  : ValueTensorBuffer = derive(json.asInstanceOf[JObject])

  final def derive(json: JObject)
  : ValueTensorBuffer = {
    val fields = json.obj.toMap
    val result = Json.toSortedMap(
      fields("banks"),
      (json: JValue) => Json.toInt(json),
      (json: JValue) => ValueTensorBank.derive(json)
    )
    apply(result)
  }

  final val empty
  : ValueTensorBuffer = apply(SortedMap.empty)

  /*
  final def zeros(layout: ParameterBufferLayout)
  : ParameterBuffer = ParameterBuffer(
    MapEx.mapValues(layout.groups)(ParameterGroup.zeros)
  )

  final def ones(layout: ParameterBufferLayout)
  : ParameterBuffer = ParameterBuffer(
    MapEx.mapValues(layout.groups)(ParameterGroup.ones)
  )
  */

  /*
  final def onesLike(reference: ParameterBuffer): ParameterBuffer = apply(
    reference.groups.map(g => ParameterGroup.onesLike(g))
  )

  final def zerosLike(reference: ParameterBuffer): ParameterBuffer = apply(
    reference.groups.map(g => ParameterGroup.zerosLike(g, 0))
  )
  */

  // TODO: Remove explicit implementations of operators and put them here as universal functions!

  /*
  implicit val canLerp
  = new OpLerp.Impl3[ParameterBuffer, ParameterBuffer, Real, ParameterBuffer] {

    override def apply(a: ParameterBuffer, b: ParameterBuffer, t: Real)
    : ParameterBuffer = a.zipGroups(b, lerp(_, _, t))

  }

  implicit val canLerpInto
  = new OpLerp.InPlaceImpl3[ParameterBuffer, ParameterBuffer, Real] {

    // TODO: Benchmark this. Pure Scala implementation could be faster!
    override def apply(a: ParameterBuffer, b: ParameterBuffer, t: Real)
    : Unit = a.foreachGroup(b, lerp.inPlace(_, _, t))

  }

  implicit val canScaleAdd
  = new scaleAdd.Impl3[ParameterBuffer, Real, ParameterBuffer, ParameterBuffer] {

    override def apply(y: ParameterBuffer, a: Real, x: ParameterBuffer)
    : ParameterBuffer = y.zipGroups(x, (y, x) => {
      val tmp = y.copy
      axpy(a, x, tmp)
      tmp
    })

  }

  implicit val canScaleAddInto
  = new scaleAdd.InPlaceImpl3[ParameterBuffer, Real, ParameterBuffer] {

    override def apply(y: ParameterBuffer, a: Real, x: ParameterBuffer)
    : Unit = x.foreachGroup(y, axpy(a, _, _))

  }

  implicit def canTraverseValues(implicit ctv: CanTraverseValues[ParameterGroup, Real])
  : CanTraverseValues[ParameterBuffer, Real]
  = new CanTraverseValues[ParameterBuffer, Real] {

    override def isTraversableAgain(from: ParameterBuffer): Boolean = true

    override def traverse(from: ParameterBuffer, fn: ValuesVisitor[Real])
    : Unit = from.foreachGroup(ctv.traverse(_, fn))

  }

  implicit def canTraverseKeyValuePairs
  : CanTraverseKeyValuePairs[ParameterBuffer, (Int, Int, Int), Real]
  = new CanTraverseKeyValuePairs[ParameterBuffer, (Int, Int, Int), Real] {

    override def isTraversableAgain(from: ParameterBuffer): Boolean = true

    override def traverse(from: ParameterBuffer,
                          fn:   KeyValuePairsVisitor[(Int, Int, Int), Real])
    : Unit = from.foreachValuePair(
      (g, s, i, v) => fn.visit((g, s, i), v)
    )

  }

  implicit def canMapValues(implicit cmv: CanMapValues[ParameterGroup, Real, Real, ParameterGroup])
  : CanMapValues[ParameterBuffer, Real, Real, ParameterBuffer]
  = new CanMapValues[ParameterBuffer, Real, Real, ParameterBuffer] {

    override def apply(from: ParameterBuffer, fn: Real => Real)
    : ParameterBuffer = from.mapParameters(fn)

  }
  */

  /*
  final def restore(stream: ObjectInputStream)
  : ParameterBuffer = {
    val builder = SortedMap.newBuilder[Int, ParameterBank]
    val noBanks = stream.readInt()
    var i = 0
    while (i < noBanks) {
      val bankNo = stream.readInt()
      val bank   = ParameterBank.restore(stream)
      builder += bankNo -> bank
      i += 1
    }
    apply(builder.result())
  }
  */

}

final class ValueTensorBufferBuilder
  extends BufferExBuilder[ValueTensorBuffer, ValueTensorBank, ValueTensor] {

  override protected def doRegister(bankNo:    Int,
                                    segmentNo: Int,
                                    item:      ValueTensor)
  : Int = {
    val bank = banks.getOrElseUpdate(bankNo, ValueTensorBankBuilder())
    bank.register(segmentNo, item)
  }

  override def result()
  : ValueTensorBuffer = ValueTensorBuffer(toSortedMap)

}

object ValueTensorBufferBuilder {

  final def apply()
  : ValueTensorBufferBuilder = new ValueTensorBufferBuilder

}
