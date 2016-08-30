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

final class RealBuffer(override val banks: SortedMap[Int, RealBank])
  extends BufferEx[RealBuffer, RealBank, Real] {

  override def toString
  : String = s"RealBuffer[${banks.size}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), banks.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[RealBuffer]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: RealBuffer =>
      banks == other.banks
    case _ =>
      false
  })


  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, RealBank])
  : RealBuffer = RealBuffer(banks)

}

object RealBuffer {

  final def apply(banks: SortedMap[Int, RealBank])
  : RealBuffer = new RealBuffer(banks)

  final val empty
  : RealBuffer = apply(SortedMap.empty)

  final def fillLike(buffer: BufferLike,
                     value:  Real)
  : RealBuffer = {
    val result = MapEx.mapValues(
      buffer.banks
    )(RealBank.fillLike(_, value))
    apply(result)
  }

  final def zeroLike(buffer: BufferLike)
  : RealBuffer = fillLike(buffer, Real.zero)

}
