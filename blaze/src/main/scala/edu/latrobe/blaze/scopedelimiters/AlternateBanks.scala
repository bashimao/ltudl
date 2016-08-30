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

package edu.latrobe.blaze.scopedelimiters

import edu.latrobe._
import edu.latrobe.blaze._
import scala.collection._
import scala.util.hashing._

final class AlternateBanks(override val builder: AlternateBanksBuilder,
                           override val scope:   NullBuffer,
                           override val seed:    InstanceSeed)
  extends IndependentScopeEx[AlternateBanksBuilder] {

  private val patternSequence
  : Array[NullBuffer] = builder.bankSequence.map(bankNo => {
    val bank = scope.banks(bankNo)
    NullBuffer.derive(bankNo, bank)
  }).toArray

  override def get(phaseNo: Long)
  : NullBuffer = {
    val index = phaseNo % patternSequence.length
    patternSequence(index.toInt)
  }

}

final class AlternateBanksBuilder
  extends IndependentScopeExBuilder[AlternateBanksBuilder] {

  override def repr
  : AlternateBanksBuilder = this

  val bankSequence
  : mutable.Buffer[Int] = mutable.Buffer.empty

  def +=(bankNo: Int)
  : AlternateBanksBuilder = {
    bankSequence += bankNo
    this
  }

  def ++=(bankNos: TraversableOnce[Int])
  : AlternateBanksBuilder = {
    bankSequence ++= bankNos
    this
  }

  override protected def doToString()
  : List[Any] = bankSequence.length :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), bankSequence.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[AlternateBanksBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: AlternateBanksBuilder =>
      bankSequence == other.bankSequence
    case _ =>
      false
  })

  override protected def doCopy()
  : AlternateBanksBuilder = AlternateBanksBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: AlternateBanksBuilder =>
        other.bankSequence.clear()
        other.bankSequence ++= bankSequence
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Instance building related.
  // ---------------------------------------------------------------------------
  override def build(source: NullBuffer,
                     seed:   InstanceSeed)
  : AlternateBanks = new AlternateBanks(this, source, seed)

}

object AlternateBanksBuilder {

  final def apply()
  : AlternateBanksBuilder = new AlternateBanksBuilder

  final def apply(bankNo0: Int)
  : AlternateBanksBuilder = apply() += bankNo0

  final def apply(bankNo0: Int, bankNos: Int*)
  : AlternateBanksBuilder = apply(bankNo0) ++= bankNos

  final def apply(bankNos: TraversableOnce[Int])
  : AlternateBanksBuilder = apply() ++= bankNos

  final val bank0
  : AlternateBanksBuilder = apply(0)

}
