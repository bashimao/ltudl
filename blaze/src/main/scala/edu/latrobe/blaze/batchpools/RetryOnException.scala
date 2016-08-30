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

package edu.latrobe.blaze.batchpools

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.io.graph._

import scala.Seq
import scala.collection._
import scala.util.hashing._

/**
  * Wraps a try-catch block around the source and tries to step over issues.
  *
  * Use this for data from unstable sources.
  */
final class RetryOnException(override val builder: RetryOnExceptionBuilder,
                             override val seed:    InstanceSeed,
                             override val source:  BatchPool)
  extends Prefetcher[RetryOnExceptionBuilder] {

  val noRetries
  : Int = builder.noRetries

  override val outputHints
  : BuildHints = inputHints

  override def draw()
  : BatchPoolDrawContext = {
    var i = 0
    do {
      try {
        return source.draw()
      }
      catch {
        case e: Exception =>
          logger.error(s"FailSafeWrapper.current exception caught -> $e")
          System.gc()
          System.runFinalization()
          i += 1
      }
    } while(i <= noRetries)
    throw new UnknownError
  }

}

final class RetryOnExceptionBuilder
  extends PrefetcherBuilder[RetryOnExceptionBuilder] {

  override def repr
  : RetryOnExceptionBuilder = this

  private var _noRetries
  : Int = 10

  def noRetries
  : Int = _noRetries

  def noRetries_=(value: Int): Unit = {
    require(value >= 0)
    _noRetries = value
  }

  def setNoRetries(value: Int)
  : RetryOnExceptionBuilder = {
    noRetries_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _noRetries :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _noRetries.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[RetryOnExceptionBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: RetryOnExceptionBuilder =>
      _noRetries == other._noRetries
    case _ =>
      false
  })

  override protected def doCopy()
  : RetryOnExceptionBuilder = RetryOnExceptionBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: RetryOnExceptionBuilder =>
        other._noRetries = _noRetries
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //   Record set construction
  // --------------------------------------------------------------------------
  override def doBuild(source: BatchPool,
                       seed:   InstanceSeed)
  : RetryOnException = new RetryOnException(this, seed, source)

}

object RetryOnExceptionBuilder {

  final def apply()
  : RetryOnExceptionBuilder = new RetryOnExceptionBuilder

  final def apply(source: BatchPoolBuilder)
  : RetryOnExceptionBuilder = apply().setSource(source)

}