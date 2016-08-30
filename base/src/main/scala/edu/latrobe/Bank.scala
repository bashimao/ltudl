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

abstract class BankLike
  extends Equatable
    with Serializable
    with JsonSerializable {

  /**
    * Should override as constructor argument!
    */
  def segments
  : SortedMap[Int, Any]

  final def noSegments
  : Int = segments.size

  def apply(segmentNo: Int)
  : Any

  def get(segmentNo: Int)
  : Option[Any]


  final def toStringEx
  : String = {
    val sb = StringBuilder.newBuilder
    MapEx.foreach(
      segments
    )((i, s) => sb ++= s"$i/$s, ")
    if (sb.nonEmpty) {
      sb.length = sb.length - 2
    }
    sb.result()
  }

  final def references
  : Set[Int] = segments.keySet

}

abstract class BankLikeCompanion
  extends JsonSerializableCompanion {

  override def derive(json: JValue)
  : BankLike

  override def derive(json: JObject)
  : BankLike

  def empty
  : BankLike

}

abstract class Bank[T]
  extends BankLike {

  override def segments
  : SortedMap[Int, T]

  final override def apply(segmentNo: Int)
  : T = segments(segmentNo)

  final override def get(segmentNo: Int)
  : Option[T] = segments.get(segmentNo)


  // ---------------------------------------------------------------------------
  //    Operations
  // ---------------------------------------------------------------------------
  final def foldLeftSegments[Z](z0: Z)
                               (fn: (Z, T) => Z)
  : Z = MapEx.foldLeftValues(
    z0,
    segments
  )(fn)

  final def foldLeftSegments[Z, U](z0: Z, other: Bank[U])
   (fn: (Z, T, U) => Z)
  : Z = MapEx.foldLeftValues(
    z0,
    segments,
    other.segments
  )(fn)

  final def foldLeftSegmentPairs[Z](z0: Z)
                                   (fn: (Z, Int, T) => Z)
  : Z = MapEx.foldLeft(
    z0,
    segments
  )(fn)

  final def foldLeftSegmentPairs[Z, U](z0: Z, other: Bank[U])
                                      (fn: (Z, Int, T, U) => Z)
  : Z = MapEx.foldLeft(
    z0,
    segments,
    other.segments
  )(fn)

  final def foreachSegment(fn: T => Unit)
  : Unit = MapEx.foreachValue(
    segments
  )(fn)

  final def foreachSegment[U](other: Bank[U])
                             (fn: (T, U) => Unit)
  : Unit = MapEx.foreachValue(
    segments,
    other.segments
  )(fn)


  final def foreachSegment[U, V](other:  Bank[U],
                                 other2: Bank[V])
                                (fn: (T, U, V) => Unit)
  : Unit = MapEx.foreachValue(
    segments,
    other.segments,
    other2.segments
  )(fn)

  final def foreachSegment[U, V, W](other:  Bank[U],
                                    other2: Bank[V],
                                    other3: Bank[W])
                                   (fn: (T, U, V, W) => Unit)
  : Unit = MapEx.foreachValue(
    segments,
    other.segments,
    other2.segments,
    other3.segments
  )(fn)

  final def foreachSegment[U, V, W, X](other:  Bank[U],
                                       other2: Bank[V],
                                       other3: Bank[W],
                                       other4: Bank[X])
                                      (fn: (T, U, V, W, X) => Unit)
  : Unit = MapEx.foreachValue(
    segments,
    other.segments,
    other2.segments,
    other3.segments,
    other4.segments
  )(fn)

  final def foreachSegmentEx[U](other: Bank[U])
                               (fn0: (T, U) => Unit,
                                fn1: T => Unit,
                                fn2: U => Unit)
  : Unit = MapEx.foreachValueEx(
    segments,
    other.segments
  )(fn0, fn1, fn2)

  final def foreachSegmentPair(fn: (Int, T) => Unit)
  : Unit = MapEx.foreach(
    segments
  )(fn)

  final def foreachSegmentPairEx[U](other: Bank[U])
                                   (fn0: (Int, T, U) => Unit,
                                    fn1: (Int, T) => Unit,
                                    fn2: (Int, U) => Unit)
  : Unit = MapEx.foreachEx(
    segments,
    other.segments
  )(fn0, fn1, fn2)

  final def mapSegments[U](fn: T => U)
  : SortedMap[Int, U] = MapEx.mapValues(
    segments
  )(fn)

  final def mapSegmentPairs[U](fn: (Int, T) => U)
  : SortedMap[Int, U] = MapEx.map(
    segments
  )(fn)

  final def mapReduceLeftSegments[U](mapFn: T => U)
                                    (reduceFn: (U, U) => U)
  : U = MapEx.mapReduceLeftValues(
    segments
  )(mapFn)(reduceFn)

  final def zipSegments[U, V](other: Bank[U])
                             (fn: (T, U) => V)
  : SortedMap[Int, V] = MapEx.zipValues(
    segments,
    other.segments
  )(fn)

  final def zipSegmentsEx[U, V](other: Bank[U])
   (fn0: (T, U) => V,
    fn1: T => V,
    fn2: U => V)
  : SortedMap[Int, V] = MapEx.mapValuesEx(
    segments,
    other.segments
  )(fn0, fn1, fn2)

  final def zipSegmentPairs[U, V](other: Bank[U])
                                 (fn: (Int, T, U) => V)
  : SortedMap[Int, V] = MapEx.zip(
    segments,
    other.segments
  )(fn)

  override protected def doToJson()
  : List[JField] = List(
    Json.field("segments", Json.derive(mapSegments(doToJson)))
  )

  protected def doToJson(segment: T)
  : JValue

}

abstract class BankCompanion[T]
  extends BankLikeCompanion {

  def apply(segments: SortedMap[Int, T])
  : BankLike

  def derive(segmentNo: Int, value: T)
  : BankLike

  def derive(segment0: (Int, T))
  : BankLike

  def derive(segment0: (Int, T), segments: (Int, T)*)
  : BankLike

}

abstract class BankEx[TBank <: BankEx[TBank, T], T]
  extends Bank[T] {

  // ---------------------------------------------------------------------------
  //    Operations
  // ---------------------------------------------------------------------------


  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  protected def doCreateView(banks: SortedMap[Int, T])
  : TBank

  final def createIntersectionView(mask: BankLike)
  : TBank = {
    val builder = SortedMap.newBuilder[Int, T]
    MapEx.foreachEx(segments, mask.segments)(
      (i, s0, s1) => builder += Tuple2(i, s0),
      (i, s0    ) => {},
      (i,     s1) => {}
    )
    doCreateView(builder.result())
  }

  final def createDifferentialView(mask: BankLike)
  : TBank = {
    val builder = SortedMap.newBuilder[Int, T]
    MapEx.foreachEx(segments, mask.segments)(
      (i, s0, s1) => {},
      (i, s0    ) => builder += Tuple2(i, s0),
      (i,     s1) => {}
    )
    doCreateView(builder.result())
  }

}

abstract class BankExCompanion[TBank <: BankEx[TBank, T], T]
  extends BankCompanion[T]
    with JsonSerializableCompanionEx[TBank] {

  override def apply(segments: SortedMap[Int, T])
  : TBank

  final override def derive(segmentNo: Int, value: T)
  : TBank = derive((segmentNo, value))

  final override def derive(segment0: (Int, T))
  : TBank = apply(SortedMap(segment0))

  final override def derive(segment0: (Int, T),
                            segments: (Int, T)*)
  : TBank = {
    val builder = SortedMap.newBuilder[Int, T]
    builder += segment0
    builder ++= segments
    apply(builder.result())
  }

  final override def derive(fields: Map[String, JValue])
  : TBank = {
    val result = Json.toSortedMap(
      fields("segments"),
      (json: JValue) => Json.toInt(json),
      (json: JValue) => doDerive(json)
    )
    apply(result)
  }

  protected def doDerive(json: JValue)
  : T

  final override val empty
  : TBank = apply(SortedMap.empty)

}

abstract class BankExBuilder[TBank <: BankEx[TBank, T], T] {

  /**
    * Used if no specific segment number has been requested.
    */
  final private var _runningSegmentNo
  : Int = 0

  final protected val segments
  : mutable.Map[Int, T] = mutable.Map.empty

  final protected def drawNextSegmentNo
  : Int = {
    _runningSegmentNo -= 1
    assume(_runningSegmentNo > Int.MinValue)

    while (segments.contains(_runningSegmentNo)) {
      _runningSegmentNo -= 1
      assume(_runningSegmentNo > Int.MinValue)
      if (logger.isWarnEnabled) {
        logger.warn(s"Automatically generated bind ID ${_runningSegmentNo} already used. This should not happen!")
      }
    }

    _runningSegmentNo
  }

  final def apply(segmentNo: Int)
  : T = segments(segmentNo)

  final def contains(segmentNo: Int)
  : Boolean = segments.contains(segmentNo)

  final def get(segmentNo: Int)
  : Option[T] = segments.get(segmentNo)

  final def noSegments
  : Int = segments.size

  final def register(segmentNo: Int, item: T): Int = {
    //require(item != null)
    if (segmentNo == 0) {
      val segmentNo = drawNextSegmentNo
      segments += Tuple2(segmentNo, item)
      segmentNo
    }
    else if (segmentNo > 0) {
      require(!segments.contains(segmentNo))
      segments += Tuple2(segmentNo, item)
      segmentNo
    }
    else {
      throw new IllegalArgumentException
    }
  }

  def result()
  : TBank

  def toMap
  : Map[Int, T] = {
    val sorted = Map.newBuilder[Int, T]
    sorted ++= segments
    sorted.result()
  }

  def toSortedMap
  : SortedMap[Int, T] = {
    val sorted = SortedMap.newBuilder[Int, T]
    sorted ++= segments
    sorted.result()
  }

}
