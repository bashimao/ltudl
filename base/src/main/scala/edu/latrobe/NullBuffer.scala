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

import scala.collection._
import scala.util.hashing._

final class NullBuffer(override val banks: SortedMap[Int, NullBank])
  extends BufferEx[NullBuffer, NullBank, Null] {

  override def toString
  : String = s"NullBuffer[${banks.size}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), banks.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LongBuffer]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: NullBuffer =>
      banks == other.banks
    case _ =>
      false
  })


  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, NullBank])
  : NullBuffer = NullBuffer(banks)

}

object NullBuffer {

  final def apply(banks: SortedMap[Int, NullBank])
  : NullBuffer = new NullBuffer(banks)

  final def derive(bankNo: Int, bank: NullBank)
  : NullBuffer = derive((bankNo, bank))

  final def derive(ref0: (Int, NullBank))
  : NullBuffer = apply(SortedMap(ref0))

  final def derive(ref0: BufferReference)
  : NullBuffer = derive(ref0.bankNo, NullBank.derive(ref0.segmentNo))

  final def derive(ref0: BufferReference, refN: BufferReference*)
  : NullBuffer = derive(ref0 :: refN.toList)

  final def derive(refN: Traversable[BufferReference])
  : NullBuffer = {
    val byBank = refN.groupBy(_.bankNo)

    val builder = SortedMap.newBuilder[Int, NullBank]
    MapEx.foreach(byBank)((i, b) => {
      val bank = NullBank.derive(b.map(_.segmentNo))
      builder += Tuple2(i, bank)
    })
    apply(builder.result())
  }

  final def derive[T](segments: SortedMap[Int, SortedMap[Int, T]])
  : NullBuffer = {
    val result = MapEx.mapValues(
      segments
    )(NullBank.derive[T])
    apply(result)
  }

  final def derive(buffer: BufferLike)
  : NullBuffer = {
    val result = MapEx.mapValues(
      buffer.banks
    )(NullBank.derive)
    apply(result)
  }

  final def deriveEx[T](segments: Seq[((Int, Int), T)])
  : NullBuffer = {
    val byBank = segments.groupBy(_._1._1)

    val builder = SortedMap.newBuilder[Int, NullBank]
    MapEx.foreach(byBank)((i, b) => {
      val bank = NullBank.derive(b.map(_._1._2))
      builder += Tuple2(i, bank)
    })
    apply(builder.result())
  }

  final val empty
  : NullBuffer = apply(SortedMap.empty)

}

final class NullBufferBuilder
  extends BufferExBuilder[NullBuffer, NullBank, Null] {

  override protected def doRegister(bankNo:    Int,
                                    segmentNo: Int,
                                    item:      Null)
  : Int = {
    val bank = banks.getOrElseUpdate(bankNo, NullBankBuilder())
    bank.register(segmentNo, item)
  }

  def +=[T <: BufferReferenceEx[T]](reference: T)
  : NullBufferBuilder = {
    register(reference, null)
    this
  }

  def ++=[T <: BufferReferenceEx[T]](references: TraversableOnce[T])
  : NullBufferBuilder = {
    references.foreach(
      +=[T]
    )
    this
  }

  override def result()
  : NullBuffer = NullBuffer(toSortedMap)

}

object NullBufferBuilder {

  final def apply()
  : NullBufferBuilder = new NullBufferBuilder

}
