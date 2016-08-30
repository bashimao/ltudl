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

import it.unimi.dsi.util._
import scala.collection._
import scala.util.hashing._

/**
  * Use XorShift128PlusRandomGenerator if 1024 star is too slow.
  */
final class PseudoRNG
//extends MersenneTwister
//extends XorShift128PlusRandomGenerator
  extends XorShift1024StarRandomGenerator
    with Serializable
    with Equatable
    with CopyableEx[PseudoRNG] {

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), ArrayEx.serialize(this).hashCode())


  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[PseudoRNG]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: PseudoRNG =>
      val bytes0 = ArrayEx.serialize(this)
      val bytes1 = ArrayEx.serialize(other)
      ArrayEx.compare(bytes0, bytes1)
    case _ =>
      false
  })

  override def copy
  : PseudoRNG = ArrayEx.deserialize(ArrayEx.serialize(this))

  // --------------------------------------------------------------------------
  //    REAL SWITCH DOUBLE
  // ---------------------------------------------------------------------------
  /*
  @inline
  def nextReal(): Real = nextDouble

  @inline
  def nextGaussianReal(): Real = nextGaussian
  */
  // -------------------------------------------------------------------------
  //    REAL SWITCH FLOAT
  // -------------------------------------------------------------------------
  ///*
  @inline
  def nextReal()
  : Real = nextFloat()

  @inline
  def nextGaussianReal()
  : Real = Real(nextGaussian())
  //*/
  // -------------------------------------------------------------------------
  //    REAL SWITCH END
  // -------------------------------------------------------------------------

  @inline
  def next[T](values: Array[T])
  : T = {
    val i = nextInt(values.length)
    values(i)
  }

  @inline
  def next[T](values: Seq[T])
  : T = {
    val i = nextInt(values.length)
    values(i)
  }

  @inline
  def nextBoolean(p: Real)
  : Boolean = {
    require(p >= Real.zero && p <= Real.one)
    nextReal() < p
  }

  @inline
  def nextReal(min: Real, max: Real)
  : Real = {
    require(min <= max)
    nextReal() * (max - min) + min
  }

  @inline
  def nextReal(range: RealRange)
  : Real = nextReal() * range.length + range.min

  @inline
  def nextGaussianReal(mu: Real, sigma: Real)
  : Real = nextGaussianReal() * sigma + mu

  // TODO: Do we still need this with the new generators?
  /*
  /**
    * Uniformly samples a long integer in [0,MAX_INT]
    */
  // TODO: Already reported to main project. https://github.com/scalanlp/breeze/issues/438
  val randPositiveInt: Rand[Int] = new Rand[Int] {
    override def draw(): Int = {
      var value = rb.generator.nextInt
      if (value < 0) {
        value -= Int.MinValue
      }
      value
    }
  }

  // TODO: I already commited this to breeze. Check and remove when we upgrade. https://github.com/scalanlp/breeze/pull/427 & https://github.com/scalanlp/breeze/pull/429
  /**
    * Uniformly samples a long integer in [0,MAX_LONG]
    */
  val randLong: Rand[Long] = new Rand[Long] {
    override def draw(): Long = {
      var value = rb.generator.nextLong
      if (value < 0L) {
        value -= Long.MinValue
      }
      value
    }
  }

  /**
    * Uniformly samples a long integer in [0,n)
    */
  def randLong(n: Long): Rand[Long] = new Rand[Long] {
    override def draw(): Long = {
      var value = rb.generator.nextLong
      if (value < 0L) {
        value -= Long.MinValue
      }
      value % n
    }
  }

  /**
    * Uniformly samples a long integer in [n,m)
    */
  def randLong(n: Long, m: Long): Rand[Long] = new Rand[Long] {
    override def draw(): Long = {
      var value = rb.generator.nextLong
      if (value < 0L) {
        value -= Long.MinValue
      }
      value % (m - n) + n
    }
  }
  */

  @inline
  def bernoulliDistribution(p: Real)
  : Distribution[Boolean] = {
    if (p == Real.pointFive) {
      new Distribution[Boolean] {

        override val isThreadSafe
        : Boolean = false

        override def sample()
        : Boolean = nextBoolean()

      }
    }
    else {
      require(p >= Real.zero && p <= Real.one)
      new Distribution[Boolean] {

        override val isThreadSafe
        : Boolean = false

        override def sample()
        : Boolean = nextReal() < p

      }
    }
  }

  @inline
  def bernoulliDistribution[T](p: Real, trueValue: T, falseValue: T)
  : Distribution[T] = {
    if (p == Real.pointFive) {
      new Distribution[T] {

        override val isThreadSafe
        : Boolean = false

        override def sample()
        : T = if (nextBoolean()) trueValue else falseValue

      }
    }
    else {
      require(p >= Real.zero && p <= Real.one)
      new Distribution[T] {

        override val isThreadSafe
        : Boolean = false

        override def sample()
        : T = if (nextReal() < p) trueValue else falseValue

      }
    }
  }

  @inline
  def uniformDistribution()
  : Distribution[Real] = {
    new Distribution[Real] {

      override val isThreadSafe
      : Boolean = false

      override def sample()
      : Real = nextReal()

    }
  }

  @inline
  def uniformDistribution(min: Real, max: Real)
  : Distribution[Real] = uniformDistribution(RealRange(min, max))

  @inline
  def uniformDistribution(range: RealRange)
  : Distribution[Real] = {
    if (range.min == Real.zero && range.max == Real.one) {
      new Distribution[Real] {

        override val isThreadSafe
        : Boolean = false

        override def sample()
        : Real = nextReal()

      }
    }
    else {
      val offset = range.min
      val scale  = range.length
      new Distribution[Real] {

        override val isThreadSafe
        : Boolean = false

        override def sample()
        : Real = nextReal() * scale + offset

      }
    }
  }

  @inline
  def gaussianDistribution()
  : Distribution[Real] = {
    new Distribution[Real] {

      override val isThreadSafe
      : Boolean = false

      override def sample()
      : Real = nextGaussianReal()

    }
  }

  @inline
  def gaussianDistribution(mu: Real, sigma: Real)
  : Distribution[Real] = {
    if (mu == Real.zero && sigma == Real.one) {
      new Distribution[Real] {

        override val isThreadSafe
        : Boolean = false

        override def sample()
        : Real = nextGaussianReal()

      }
    }
    else {
      new Distribution[Real] {

        override val isThreadSafe
        : Boolean = false

        override def sample()
        : Real = nextGaussianReal(mu, sigma)

      }
    }
  }

}

/**
  * Enhanced version of the Rand object that supports seeding.
  */
object PseudoRNG {

  final val default
  : PseudoRNG = apply()

  final def apply()
  : PseudoRNG = new PseudoRNG

  final def apply(seed: Long)
  : PseudoRNG = {
    val rng = apply()
    rng.setSeed(seed)
    rng
  }

}