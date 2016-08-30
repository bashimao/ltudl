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

package edu.latrobe.blaze

import edu.latrobe._
import scala.util.hashing._

/**
  * Train mode is used by optimizers.
  *
  * @param phaseNo For local optimizers the runNo. For other optimizers this
  *                might be a different value.
  */
final class Training(val phaseNo:               Long,
                     override val reproducible: Boolean)
  extends Mode {
  require(phaseNo >= 0L)

  override def toString
  : String = s"Training[$phaseNo, $reproducible]"

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, phaseNo.hashCode())
    tmp = MurmurHash3.mix(tmp, reproducible.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Training]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Training =>
      phaseNo      == other.phaseNo &&
      reproducible == other.reproducible
    case _ =>
      false
  })

  override def supportsBackpropagation
  : Boolean = true

}

object Training {

  final def apply(phaseNo: Long)
  : Training = apply(phaseNo, reproducible = false)

  final def apply(phaseNo: Long, reproducible: Boolean)
  : Training = new Training(phaseNo, reproducible)

  final def reproducible(phaseNo: Long)
  : Mode = apply(phaseNo, reproducible = true)

}