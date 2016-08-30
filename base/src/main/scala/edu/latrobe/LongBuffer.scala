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

final class LongBuffer(override val banks: SortedMap[Int, LongBank])
  extends BufferEx[LongBuffer, LongBank, Long] {

  override def toString
  : String = s"LongBuffer[${banks.size}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), banks.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LongBuffer]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: LongBuffer =>
      banks == other.banks
    case _ =>
      false
  })

  def +(value: Long)
  : LongBuffer = {
    val result = mapBanks(
      _ + value
    )
    LongBuffer(result)
  }

  def +(other: LongBuffer)
  : LongBuffer = {
    val result = zipBanksEx(
      other
    )((b0, b1) => b0 + b1, b0 => b0, b1 => b1)
    LongBuffer(result)
  }

  def -(value: Long)
  : LongBuffer = {
    val result = mapBanks(
      _ - value
    )
    LongBuffer(result)
  }

  def -(other: LongBuffer)
  : LongBuffer = {
    val result = zipBanksEx(
      other
    )((b0, b1) => b0 - b1, b0 => b0, b1 => b1)
    LongBuffer(result)
  }


  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, LongBank])
  : LongBuffer = LongBuffer(banks)

}

object LongBuffer {

  final def apply(banks: SortedMap[Int, LongBank])
  : LongBuffer = new LongBuffer(banks)

  final val empty
  : LongBuffer = apply(SortedMap.empty)

  final def fillLike(buffer: BufferLike,
                     value:  Long)
  : LongBuffer = {
    val result = MapEx.mapValues(
      buffer.banks
    )(LongBank.fillLike(_, value))
    apply(result)
  }

  final def zeroLike(buffer: BufferLike)
  : LongBuffer = fillLike(buffer, 0L)

}
