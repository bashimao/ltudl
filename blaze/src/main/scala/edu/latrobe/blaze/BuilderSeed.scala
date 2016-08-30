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

import edu.latrobe._
import scala.util.hashing._

/**
 * The builder part of a random seed. This helper class is mainly there to keep
 * the rest of the interface clean. (otherwise we would have to modify hundreds
 * of files when adding or removing a seed source)
 *
 * Immutable!
 */
final class BuilderSeed(val baseSeed:        Int,
                        val agentFactor:    Int,
                        val partitionFactor: Int,
                        val extraFactor:     Int,
                        val temporalFactor:  Int,
                        val machineFactor:   Int)
  extends Serializable
    with Equatable {

  override def toString: String = {
    s"BuilderSeed($baseSeed, $agentFactor, $partitionFactor, $extraFactor, $temporalFactor, $machineFactor)"
  }

  override def canEqual(that: Any): Boolean = that.isInstanceOf[BuilderSeed]

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, baseSeed.hashCode())
    tmp = MurmurHash3.mix(tmp, agentFactor.hashCode())
    tmp = MurmurHash3.mix(tmp, partitionFactor.hashCode())
    tmp = MurmurHash3.mix(tmp, extraFactor.hashCode())
    tmp = MurmurHash3.mix(tmp, temporalFactor.hashCode())
    tmp = MurmurHash3.mix(tmp, machineFactor.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: BuilderSeed =>
      baseSeed        == other.baseSeed        &&
      agentFactor    == other.agentFactor    &&
      partitionFactor == other.partitionFactor &&
      extraFactor     == other.extraFactor     &&
      temporalFactor  == other.temporalFactor  &&
      machineFactor   == other.machineFactor
    case _ =>
      false
  })

  def setBaseSeed(baseSeed: Int): BuilderSeed = BuilderSeed(
    baseSeed,
    agentFactor, partitionFactor, extraFactor,
    temporalFactor, machineFactor
  )

  def setAgentFactor(threadFactor: Int): BuilderSeed = BuilderSeed(
    baseSeed,
    threadFactor, partitionFactor, extraFactor,
    temporalFactor, machineFactor
  )

  def setPartitionFactor(partitionFactor: Int): BuilderSeed = BuilderSeed(
    baseSeed,
    agentFactor, partitionFactor, extraFactor,
    temporalFactor, machineFactor
  )

  def setExtraFactor(extraFactor: Int): BuilderSeed = BuilderSeed(
    baseSeed,
    agentFactor, partitionFactor, extraFactor,
    temporalFactor, machineFactor
  )

  def setTemporalFactor(temporalFactor: Int): BuilderSeed = BuilderSeed(
    baseSeed,
    agentFactor, partitionFactor, extraFactor,
    temporalFactor, machineFactor
  )

  def setMachineFactor(machineFactor: Int): BuilderSeed = BuilderSeed(
    baseSeed,
    agentFactor, partitionFactor, extraFactor,
    temporalFactor, machineFactor
  )

}

object BuilderSeed {

  final def apply(baseSeed:        Int = PseudoRNG.default.nextInt(),
                  threadFactor:    Int = 1,
                  partitionFactor: Int = 1,
                  extraFactor:     Int = 1,
                  temporalFactor:  Int = 1,
                  machineFactor:   Int = 1)
  : BuilderSeed = new BuilderSeed(
    baseSeed,
    threadFactor,
    partitionFactor,
    extraFactor,
    temporalFactor,
    machineFactor
  )

  final def reproducible()
  : BuilderSeed = apply(
    0,
    1,
    1,
    1,
    0,
    1
  )

}


