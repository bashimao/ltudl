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

final class MeanBuffer(override val banks: SortedMap[Int, MeanBank])
  extends BufferEx[MeanBuffer, MeanBank, Mean] {

  override def toString
  : String = s"MeanBuffer[${banks.size}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), banks.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[MeanBuffer]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: MeanBuffer =>
      banks == other.banks
    case _ =>
      false
  })


  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, MeanBank])
  : MeanBuffer = MeanBuffer(banks)

}

object MeanBuffer {

  final def apply(banks: SortedMap[Int, MeanBank])
  : MeanBuffer = new MeanBuffer(banks)

  final val empty
  : MeanBuffer = apply(SortedMap.empty)

}
