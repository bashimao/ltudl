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

import java.net._
import java.util.UUID
import edu.latrobe._
import edu.latrobe.io._
import edu.latrobe.time._
import scala.util.hashing._

abstract class Instance
  extends Closable {

  /**
    * Must be overwritten with a constructor argument.
    */
  def builder
  : InstanceBuilder

  /**
    * Must be overwritten with a constructor argument.
    */
  def seed
  : InstanceSeed

  final val uniqueID
  : UUID = builder.id

  final override lazy val toString
  : String = builder.toString

  final val builderSeed
  : BuilderSeed = builder.seed

  final private var _rng
  : PseudoRNG = _

  @transient
  final lazy val rng
  : PseudoRNG = {
    if (_rng == null) {
      var s = hashSeed
      s = MurmurHash3.mix(s, builderSeed.agentFactor     * seed.agentNo)
      s = MurmurHash3.mix(s, builderSeed.partitionFactor * seed.partitionNo)
      s = MurmurHash3.mix(s, builderSeed.extraFactor     * seed.extraSeed)
      if (builderSeed.temporalFactor != 0) {
        val now = Timestamp.now()
        s = MurmurHash3.mix(s, builderSeed.temporalFactor * now.hashCode())
      }
      if (builderSeed.machineFactor != 0) {
        s = MurmurHash3.mix(s, builderSeed.machineFactor * Host.name.hashCode())
        val nis = NetworkInterface.getNetworkInterfaces
        while (nis.hasMoreElements) {
          val tmp = nis.nextElement.getHardwareAddress
          if (tmp != null) {
            s = MurmurHash3.mix(s, ArrayEx.hashCode(tmp))
          }
        }
      }
      _rng = PseudoRNG(s)
    }
    _rng
  }


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  /**
    * @return After the state has been retrieved the object that created it must
    *         not be used anymore because it may share mutable objects with its
    *         creator.
    */
  def state
  : InstanceState = PseudoRNGState(ArrayEx.serialize(rng))

  /**
    * However, restoring a state is always safe. Hence, the state is merely
    * copied into the object.
    */
  def restoreState(state: InstanceState)
  : Unit = {
    require(_rng == null)
    state match {
      case state: PseudoRNGState =>
        _rng = ArrayEx.deserialize(state.rng)
      case _ =>
        throw new MatchError(state)
    }
  }

}

abstract class InstanceBuilder
  extends Equatable
    with Copyable
    with Serializable {

  def repr
  : InstanceBuilder

  final val id
  : UUID = UUID.randomUUID()

  final private var _seed
  : BuilderSeed = BuilderSeed()

  final def seed
  : BuilderSeed = _seed

  final def seed_=(value: BuilderSeed)
  : Unit = {
    require(value != null)
    _seed = value
  }

  def setSeed(value: BuilderSeed)
  : InstanceBuilder

  final def shortName
  : String = {
    val simpleName = getClass.getSimpleName
    if (simpleName.endsWith("Builder")) {
      simpleName.substring(0, simpleName.length - "Builder".length)
    }
    else {
      // Something is wrong if we reach here.
      simpleName
    }
  }

  // This can be a horrible thing to do if toString has not been properly overloaded.
  // Hence we enforce doing so.
  final override def toString
  : String = toString("[", "]")

  final def toString(infix: String, postfix: String)
  : String = {
    val builder = StringBuilder.newBuilder

    // Prefix short name.
    builder ++= shortName

    // Parameters for display.
    val values = doToString()
    if (values.nonEmpty) {
      builder ++= infix
      var i = 0
      values.foreach(value => {
        builder ++= s"$value, "
        i += 1
      })
      if (i > 0) {
        builder.length = builder.length - 2
      }
      builder ++= postfix
    }

    builder.result()
  }

  /**
    * Fetches important parameters to print.
    */
  // TODO: Have to find some cheap labour to convert this into a map.
  protected def doToString()
  : List[Any] = Nil

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _seed.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: InstanceBuilder =>
      _seed == other._seed
    case _ =>
      false
  })

  override def copy
  : InstanceBuilder

  def copyTo(other: InstanceBuilder)
  : Unit = {
    other match {
      case other: InstanceBuilder =>
        other._seed = _seed
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  def permuteSeeds(fn: BuilderSeed => BuilderSeed)
  : InstanceBuilder

  protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = seed_=(fn(_seed))

}

abstract class InstanceBuilder0
  extends InstanceBuilder {

  override def copy
  : InstanceBuilder0

  def build()
  : Instance

}

/**
 * Non-serializable object with a state that can be restored.
 */
abstract class InstanceEx[TBuilder <: InstanceExBuilder[_]]
  extends Instance {

  /**
   * Must be overwritten with a constructor argument.
   */
  override def builder
  : TBuilder

}

/**
 * A builder with a seed.
 */
abstract class InstanceExBuilder[TThis <: InstanceExBuilder[_]]
  extends InstanceBuilder
    with CopyableEx[TThis] {

  override def repr
  : TThis

  final override def copy
  : TThis = {
    val result = doCopy()
    copyTo(result)
    result
  }

  /**
    * First step of the copy process.
    */
  protected def doCopy()
  : TThis

  final override def setSeed(value: BuilderSeed)
  : TThis = {
    seed_=(value)
    repr
  }


  // ---------------------------------------------------------------------------
  //    Mutable variables and permutation.
  // ---------------------------------------------------------------------------
  final override def permuteSeeds(fn: BuilderSeed => BuilderSeed)
  : TThis = {
    doPermuteSeeds(fn)
    repr
  }

}

abstract class InstanceExBuilder0[TThis <: InstanceExBuilder0[_, _], T <: InstanceEx[TThis]]
  extends InstanceExBuilder[TThis] {

  final def build()
  : T = build(InstanceSeed.default)

  def build(seed: InstanceSeed)
  : T

}

abstract class InstanceExBuilder1[TThis <: InstanceExBuilder1[_, _, _], T <: InstanceEx[TThis], U]
  extends InstanceExBuilder[TThis] {

  final def build(context: U)
  : T = build(context, InstanceSeed.default)

  def build(context: U, seed: InstanceSeed)
  : T

}

abstract class InstanceExBuilder2[TThis <: InstanceExBuilder2[_, _, _, _], T <: InstanceEx[TThis], U, V]
  extends InstanceExBuilder[TThis] {

  final def build(context0: U, context1: V)
  : T = build(context0, context1, InstanceSeed.default)

  def build(context0: U, context1: V, seed: InstanceSeed)
  : T

}

abstract class InstanceExBuilder3[TThis <: InstanceExBuilder3[_, _, _, _, _], T <: InstanceEx[TThis], U, V, W]
  extends InstanceExBuilder[TThis] {

  final def build(context0: U, context1: V, context2: W)
  : T = build(context0, context1, context2, InstanceSeed.default)

  def build(context0: U, context1: V, context2: W, seed: InstanceSeed)
  : T

}


