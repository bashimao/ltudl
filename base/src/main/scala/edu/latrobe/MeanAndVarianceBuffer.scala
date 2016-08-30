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

final class MeanAndVarianceBuffer(override val banks: SortedMap[Int, MeanAndVarianceBank])
  extends BufferEx[MeanAndVarianceBuffer, MeanAndVarianceBank, MeanAndVariance] {

  override def toString
  : String = s"MeanAndVarianceBuffer[${banks.size}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), banks.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[MeanAndVarianceBuffer]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: MeanAndVarianceBuffer =>
      banks == other.banks
    case _ =>
      false
  })


  // ---------------------------------------------------------------------------
  //    Conversion
  // ---------------------------------------------------------------------------
  override protected def doCreateView(banks: SortedMap[Int, MeanAndVarianceBank])
  : MeanAndVarianceBuffer = MeanAndVarianceBuffer(banks)

}

object MeanAndVarianceBuffer {

  final def apply(banks: SortedMap[Int, MeanAndVarianceBank])
  : MeanAndVarianceBuffer = new MeanAndVarianceBuffer(banks)

  final val empty
  : MeanAndVarianceBuffer = apply(SortedMap.empty)

}
