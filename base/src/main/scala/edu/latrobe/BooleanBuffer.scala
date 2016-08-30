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

final class BooleanBuffer(override val banks: SortedMap[Int, BooleanBank])
  extends BufferEx[BooleanBuffer, BooleanBank, Boolean] {

  override def toString
  : String = s"BooleanBuffer[${banks.size}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), banks.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[BooleanBuffer]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: BooleanBuffer =>
      banks == other.banks
    case _ =>
      false
  })

  def &&(value: Boolean)
  : BooleanBuffer = {
    val result = mapBanks(
      _ && value
    )
    BooleanBuffer(result)
  }

  def ||(value: Boolean)
  : BooleanBuffer = {
    val result = mapBanks(
      _ || value
    )
    BooleanBuffer(result)
  }


  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, BooleanBank])
  : BooleanBuffer = BooleanBuffer(banks)

}

object BooleanBuffer {

  final def apply(banks: SortedMap[Int, BooleanBank])
  : BooleanBuffer = new BooleanBuffer(banks)

  final def derive(bankNo: Int, bank: BooleanBank)
  : BooleanBuffer = derive((bankNo, bank))

  final def derive(bank0: (Int, BooleanBank))
  : BooleanBuffer = apply(SortedMap(bank0))

  final def derive(reference0: SimpleBufferReference, value: Boolean)
  : BooleanBuffer = derive(
    reference0.bankNo,
    BooleanBank.derive(
      reference0.segmentNo,
      value
    )
  )

  final def deriveTrue(reference0: SimpleBufferReference)
  : BooleanBuffer = derive(reference0, value = true)

  final def deriveFalse(reference0: SimpleBufferReference)
  : BooleanBuffer = derive(reference0, value = false)

  final val empty
  : BooleanBuffer = apply(SortedMap.empty)

  final def fillLike(buffer: BufferLike,
                     value:  Boolean)
  : BooleanBuffer = {
    val result = MapEx.mapValues(
      buffer.banks
    )(BooleanBank.fillLike(_, value))
    apply(result)
  }

  final def trueLike(buffer: BufferLike)
  : BooleanBuffer = fillLike(buffer, value = true)

  final def falseLike(buffer: BufferLike)
  : BooleanBuffer = fillLike(buffer, value = false)

}
