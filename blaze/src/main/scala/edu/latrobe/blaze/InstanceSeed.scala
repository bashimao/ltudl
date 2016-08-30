/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2014 Matthias Langer (t3l@threelights.de)
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
 */

package edu.latrobe.blaze

import edu.latrobe.Equatable

import scala.util.hashing._

/**
 * The instance part of a random seed. This helper class is mainly there to keep
 * the rest of the interface clean. (otherwise we would have to modify hundreds
 * of files when adding or removing a seed source)
 */
final class InstanceSeed(val agentNo:    Int,
                         val partitionNo: Int,
                         val extraSeed:   Int)
  extends Serializable
    with Equatable {

  override def toString
  : String = s"InstanceSeed[$agentNo, $partitionNo, $extraSeed]"

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, agentNo.hashCode())
    tmp = MurmurHash3.mix(tmp, partitionNo.hashCode())
    tmp = MurmurHash3.mix(tmp, extraSeed.hashCode())
    tmp
  }

  override def canEqual(that: Any): Boolean = that.isInstanceOf[InstanceSeed]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: InstanceSeed =>
      agentNo    == other.agentNo    &&
      partitionNo == other.partitionNo &&
      extraSeed   == other.extraSeed
    case _ =>
      false
  })

  def withAgentNo(agentNo: Int)
  : InstanceSeed = InstanceSeed(agentNo, partitionNo, extraSeed)

  def withPartitionNo(partitionNo: Int)
  : InstanceSeed = InstanceSeed(agentNo, partitionNo, extraSeed)

  def withExtraSeed(extraSeed: Int)
  : InstanceSeed = InstanceSeed(agentNo, partitionNo, extraSeed)

}

object InstanceSeed {

  final val default
  : InstanceSeed = apply(-1, -1)

  final def apply(agentNo: Int, partitionNo: Int)
  : InstanceSeed = apply(agentNo, partitionNo, 0)

  final def apply(agentNo: Int, partitionNo: Int, extraSeed: Int)
  : InstanceSeed = new InstanceSeed(agentNo, partitionNo, extraSeed)

}
