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

abstract class BufferLike
  extends Equatable
    with Serializable
    with JsonSerializable {

  def banks
  : SortedMap[Int, BankLike]

  final def noBanks
  : Int = banks.size

  final def noSegments
  : Int = MapEx.foldLeftValues(
    0,
    banks
  )(_ + _.noSegments)

  def apply(bankNo: Int)
  : Any

  def apply(bankNo: Int, segmentNo: Int)
  : Any

  def apply(reference: BufferReference)
  : Any

  def get(bankNo: Int)
  : Option[Any]

  def get(bankNo: Int, segmentNo: Int)
  : Option[Any]

  def get(reference: BufferReference)
  : Option[Any]

  final def toStringEx
  : String = {
    val sb = StringBuilder.newBuilder
    MapEx.foreach(
      banks
    )((i, b) => sb ++= s"$i -> {${b.toStringEx}}, ")
    if (sb.nonEmpty) {
      sb.length = sb.length - 2
    }
    sb.result()
  }

  final def references
  : Array[SimpleBufferReference] = {
    val builder = Array.newBuilder[SimpleBufferReference]
    MapEx.foreach(
      banks
    )((i, s) => s.references.foreach(j => builder += SimpleBufferReference(i, j)))
    builder.result()
  }

  final def isCompatibleWith(other: BufferLike)
  : Boolean = ArrayEx.compare(
    references,
    other.references
  )

}

abstract class Buffer[TBank <: Bank[T], T]
  extends BufferLike {

  override def banks
  : SortedMap[Int, TBank]

  final override def apply(bankNo: Int)
  : TBank = banks(bankNo)

  final override def apply(bankNo: Int, segmentNo: Int)
  : T = banks(bankNo).segments(segmentNo)

  final override def apply(reference: BufferReference)
  : T = apply(reference.bankNo, reference.segmentNo)

  final override def get(bankNo: Int)
  : Option[TBank] = banks.get(bankNo)

  final override def get(bankNo: Int, segmentNo: Int)
  : Option[T] = {
    val bank = get(bankNo)
    if (bank.isDefined) {
      bank.get.get(segmentNo)
    }
    else {
      None
    }
  }

  final override def get(reference: BufferReference)
  : Option[T] = get(reference.bankNo, reference.segmentNo)


  // ---------------------------------------------------------------------------
  //    Operations
  // ---------------------------------------------------------------------------
  final def foldLeftBanks[Z](z0: Z)
                            (fn: (Z, TBank) => Z)
  : Z = MapEx.foldLeftValues(
    z0,
    banks
  )(fn)

  final def foldLeftBanks[
  Z,
  UBank <: Bank[U],
  U
  ](z0: Z, other: Buffer[UBank, U])
   (fn: (Z, TBank, UBank) => Z)
  : Z = MapEx.foldLeftValues(
    z0,
    banks,
    other.banks
  )(fn)

  final def foldLeftBankPairs[Z](z0: Z)
                                (fn: (Z, Int, TBank) => Z)
  : Z = MapEx.foldLeft(
    z0,
    banks
  )(fn)

  final def foldLeftBankPairs[
  Z,
  UBank <: Bank[U],
  U
  ](z0: Z, other: Buffer[UBank, U])
   (fn: (Z, Int, TBank, UBank) => Z)
  : Z = MapEx.foldLeft(
    z0,
    banks,
    other.banks
  )(fn)

  final def foldLeftSegments[Z](z0: Z)
                               (fn: (Z, T) => Z)
  : Z = foldLeftBanks(
    z0
  )((z0, a) => a.foldLeftSegments(z0)(fn))

  final def foldLeftSegments[
  Z,
  UBank <: Bank[U],
  U](z0: Z, other: Buffer[UBank, U])
    (fn: (Z, T, U) => Z)
  : Z = foldLeftBanks(
    z0,
    other
  )((z0, a, b) => a.foldLeftSegments(z0, b)(fn))

  final def foldLeftSegmentPairs[Z](z0: Z)
                                   (fn: (Z, Int, Int, T) => Z)
  : Z = foldLeftBankPairs(
    z0
  )((z0, i, a) => a.foldLeftSegmentPairs(z0)(fn(_, i, _, _)))

  final def foldLeftSegmentPairs[
  Z,
  UBank <: Bank[U],
  U
  ](z0: Z, other: Buffer[UBank, U])
   (fn: (Z, Int, Int, T, U) => Z)
  : Z = foldLeftBankPairs(
    z0,
    other
  )((z0, i, a, b) => a.foldLeftSegmentPairs(z0, b)(fn(_, i, _, _, _)))

  final def foreachBank(fn: TBank => Unit)
  : Unit = MapEx.foreachValue(
    banks
  )(fn)

  final def foreachBank[
  UBank <: Bank[U], U
  ](other: Buffer[UBank, U])
   (fn: (TBank, UBank) => Unit)
  : Unit = MapEx.foreachValue(
    banks,
    other.banks
  )(fn)

  final def foreachBank[
  UBank <: Bank[U], U,
  VBank <: Bank[V], V
  ](other:  Buffer[UBank, U],
    other2: Buffer[VBank, V])
   (fn: (TBank, UBank, VBank) => Unit)
  : Unit = MapEx.foreachValue(
    banks,
    other.banks,
    other2.banks
  )(fn)

  final def foreachBank[
  UBank <: Bank[U], U,
  VBank <: Bank[V], V,
  WBank <: Bank[W], W
  ](other:  Buffer[UBank, U],
    other2: Buffer[VBank, V],
    other3: Buffer[WBank, W])
   (fn: (TBank, UBank, VBank, WBank) => Unit)
  : Unit = MapEx.foreachValue(
    banks,
    other.banks,
    other2.banks,
    other3.banks
  )(fn)

  final def foreachBank[
  UBank <: Bank[U], U,
  VBank <: Bank[V], V,
  WBank <: Bank[W], W,
  XBank <: Bank[X], X
  ](other:  Buffer[UBank, U],
    other2: Buffer[VBank, V],
    other3: Buffer[WBank, W],
    other4: Buffer[XBank, X])
   (fn: (TBank, UBank, VBank, WBank, XBank) => Unit)
  : Unit = MapEx.foreachValue(
    banks,
    other.banks,
    other2.banks,
    other3.banks,
    other4.banks
  )(fn)

  final def foreachBankEx[
  UBank <: Bank[U],
  U
  ](other: Buffer[UBank, U])
   (fn0: (TBank, UBank) => Unit,
    fn1: TBank => Unit,
    fn2: UBank => Unit)
  : Unit = MapEx.foreachValueEx(
    banks,
    other.banks
  )(fn0, fn1, fn2)

  final def foreachBankPairEx[
  UBank <: Bank[U],
  U
  ](other: Buffer[UBank, U])
   (fn0: (Int, TBank, UBank) => Unit,
    fn1: (Int, TBank) => Unit,
    fn2: (Int, UBank) => Unit)
  : Unit = MapEx.foreachEx(
    banks,
    other.banks
  )(fn0, fn1, fn2)

  final def foreachBankPair(fn: (Int, TBank) => Unit)
  : Unit = MapEx.foreach(
    banks
  )(fn)

  final def foreachSegment(fn: T => Unit)
  : Unit = foreachBank(
    _.foreachSegment(fn)
  )

  final def foreachSegment[
  UBank <: Bank[U],
  U
  ](other: Buffer[UBank, U])
   (fn: (T, U) => Unit)
  : Unit = foreachBank(
    other
  )(_.foreachSegment(_)(fn))

  /**
    * Prefer using this variant with a statically compiled function because it
    * tends to be inlined better.
    */
  final def foreachSegment[
  UBank <: Bank[U], U,
  VBank <: Bank[V], V
  ](other:  Buffer[UBank, U],
    other2: Buffer[VBank, V])
   (fn: (T, U, V) => Unit)
  : Unit = foreachBank(
    other,
    other2
  )(_.foreachSegment(_, _)(fn))

  final def foreachSegment[
  UBank <: Bank[U], U,
  VBank <: Bank[V], V,
  WBank <: Bank[W], W
  ](other:  Buffer[UBank, U],
    other2: Buffer[VBank, V],
    other3: Buffer[WBank, W])
   (fn: (T, U, V, W) => Unit)
  : Unit = foreachBank(
    other,
    other2,
    other3
  )(_.foreachSegment(_, _, _)(fn))


  final def foreachSegment[
  UBank <: Bank[U], U,
  VBank <: Bank[V], V,
  WBank <: Bank[W], W,
  XBank <: Bank[X], X
  ](other:  Buffer[UBank, U],
    other2: Buffer[VBank, V],
    other3: Buffer[WBank, W],
    other4: Buffer[XBank, X])
   (fn: (T, U, V, W, X) => Unit)
  : Unit = foreachBank(
    other,
    other2,
    other3,
    other4
  )(_.foreachSegment(_, _, _, _)(fn))

  final def foreachSegmentEx[
  UBank <: Bank[U],
  U
  ](other: Buffer[UBank, U])
   (fn0: (T, U) => Unit,
    fn1: T => Unit,
    fn2: U => Unit)
  : Unit = foreachBankEx(other)(
    _.foreachSegmentEx(
      _
    )(fn0, fn1, fn2),
    _.foreachSegment(fn1),
    _.foreachSegment(fn2)
  )

  final def foreachSegmentPair(fn: (Int, Int, T) => Unit)
  : Unit = foreachBankPair(
    (i, b) => b.foreachSegmentPair(fn(i, _, _))
  )

  final def mapBanks[U](fn: TBank => U)
  : SortedMap[Int, U] = MapEx.mapValues(
    banks
  )(fn)

  final def mapBankPairs[V](fn: (Int, TBank) => V)
  : SortedMap[Int, V] = MapEx.map(
    banks
  )(fn)

  final def mapReduceLeftBanks[V](mapFn: TBank => V)
                                 (reduceFn: (V, V) => V)
  : V = MapEx.mapReduceLeftValues(
    banks
  )(mapFn)(reduceFn)

  final def mapSegments[U](fn: T => U)
  : SortedMap[(Int, Int), U] = {
    val builder = SortedMap.newBuilder[(Int, Int), U]
    foreachSegmentPair(
      (i, j, s) => builder += Tuple2((i, j), fn(s))
    )
    builder.result()
  }

  final def segments
  : SortedMap[(Int, Int), T] = {
    val builder = SortedMap.newBuilder[(Int, Int), T]
    foreachSegmentPair(
      (i, j, s) => builder += Tuple2((i, j), s)
    )
    builder.result()
  }

  final def zipBanks[
  UBank <: Bank[U],
  U,
  V
  ](other: Buffer[UBank, U])
   (fn: (TBank, UBank) => V)
  : SortedMap[Int, V] = MapEx.zipValues(
    banks,
    other.banks
  )(fn)

  final def zipBanksEx[
  UBank <: Bank[U],
  U,
  V
  ](other: Buffer[UBank, U])
   (fn0: (TBank, UBank) => V,
    fn1: TBank => V,
    fn2: UBank => V)
  : SortedMap[Int, V] = MapEx.mapValuesEx(
    banks,
    other.banks
  )(fn0, fn1, fn2)

  final def zipBankPairs[
  UBank <: Bank[U],
  U,
  V
  ](other: Buffer[UBank, U])
   (fn: (Int, TBank, UBank) => V)
  : SortedMap[Int, V] = MapEx.zip(
    banks,
    other.banks
  )(fn)


  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  final override protected def doToJson()
  : List[JField] = List(
    Json.field("banks", Json.derive(mapBanks(_.toJson)))
  )

}

abstract class BufferEx[TBuffer <: BufferEx[TBuffer, TBank, T], TBank <: BankEx[TBank, T], T]
  extends Buffer[TBank, T] {

  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  protected def doCreateView(banks: SortedMap[Int, TBank])
  : TBuffer

  final def createIntersectionView(mask: BufferLike)
  : TBuffer = {
    val builder = SortedMap.newBuilder[Int, TBank]
    MapEx.foreachEx(banks, mask.banks)(
      (i, b0, b1) => builder += Tuple2(i, b0.createIntersectionView(b1)),
      (i, b0    ) => {},
      (i,     b1) => {}
    )
    doCreateView(builder.result())
  }

  final def createDifferentialView(mask: BufferLike)
  : TBuffer = {
    val builder = SortedMap.newBuilder[Int, TBank]
    MapEx.foreachEx(banks, mask.banks)(
      (i, b0, b1) => builder += Tuple2(i, b0.createDifferentialView(b1)),
      (i, b0    ) => b0,
      (i,     b1) => {}
    )
    doCreateView(builder.result())
  }

}

abstract class BufferExBuilder[TBuffer <: BufferEx[TBuffer, TBank, T], TBank <: BankEx[TBank, T], T] {

  final protected val banks
  : mutable.Map[Int, BankExBuilder[TBank, T]] = mutable.Map.empty

  final def apply(bankNo: Int, segmentNo: Int)
  : T = banks(bankNo)(segmentNo)

  final def apply(reference: LabeledBufferReference)
  : T = apply(reference.bankNo, reference.segmentNo)

  final def contains(reference: LabeledBufferReference)
  : Boolean = {
    val bank = banks.get(reference.bankNo)
    if (bank.isDefined) {
      bank.get.contains(reference.segmentNo)
    }
    else {
      false
    }
  }

  final def get(reference: LabeledBufferReference)
  : Option[T] = {
    val bank = banks.get(reference.bankNo)
    if (bank.isDefined) {
      bank.get.get(reference.segmentNo)
    }
    else {
      None
    }
  }

  final def noBanks
  : Int = banks.size

  final def noSegments
  : Int = MapEx.foldLeftValues(0, banks)(_ + _.noSegments)

  final def register[TRef <: BufferReferenceEx[TRef]](reference: TRef, item: T)
  : TRef = {
    val segmentNo = doRegister(
      reference.bankNo,
      reference.segmentNo,
      item
    )
    reference.derive(segmentNo)
  }

  protected def doRegister(bankNo: Int, segmentNo: Int, item: T)
  : Int

  def result()
  : TBuffer

  final def toMap
  : Map[Int, TBank] = {
    val sorted = Map.newBuilder[Int, TBank]
    MapEx.foreach(
      banks
    )((bankNo, group) => sorted += Tuple2(bankNo, group.result()))
    sorted.result()
  }

  final def toSortedMap
  : SortedMap[Int, TBank] = {
    val sorted = SortedMap.newBuilder[Int, TBank]
    MapEx.foreach(
      banks
    )((i, b) => sorted += Tuple2(i, b.result()))
    sorted.result()
  }

}
