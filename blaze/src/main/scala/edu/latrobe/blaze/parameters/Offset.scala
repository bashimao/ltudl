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

package edu.latrobe.blaze.parameters

import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

final class Offset(override val builder: OffsetBuilder,
                   override val name:    String,
                   override val seed:    InstanceSeed)
  extends DependentParameter[OffsetBuilder] {

  val amount
  : Real = builder.amount

  override def get(phaseNo: Long)
  : Real = super.get(phaseNo) + amount

}

final class OffsetBuilder
  extends DependentParameterBuilder[OffsetBuilder] {

  override def repr
  : OffsetBuilder = this

  var amount
  : Real = Real.one

  def setAmount(value: Real)
  : OffsetBuilder = {
    amount_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = f"$amount%.4g" :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), amount.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[OffsetBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: OffsetBuilder =>
      amount == other.amount
    case _ =>
      false
  })

  override protected def doCopy()
  : OffsetBuilder = OffsetBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: OffsetBuilder =>
        other.amount = amount
      case _ =>
    }
  }

  override def build(name: String, seed: InstanceSeed)
  : Offset = new Offset(this, name, seed)

}

object OffsetBuilder {

  final def apply()
  : OffsetBuilder = new OffsetBuilder

  final def apply(amount: Real)
  : OffsetBuilder = apply().setAmount(amount)

}