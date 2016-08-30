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
import edu.latrobe.time._
import java.util.concurrent.{LinkedBlockingQueue, TimeUnit}
import java.util.concurrent.atomic.AtomicBoolean
import scala.util.hashing._

/**
  * An advanced version of the asynchronous prefetcher that uses a dedicated
  * thread and can prefetch several batches.
  */
final class AdvancedPrefetcher(override val builder: AdvancedPrefetcherBuilder,
                               override val seed:    InstanceSeed,
                               override val source:  BatchPool)
  extends Prefetcher[AdvancedPrefetcherBuilder] {

  override val outputHints
  : BuildHints = inputHints

  private val queueLength
  : Int = builder.queueLength

  private val queueOfferTimeout
  : Long = builder.queueOfferTimeout.getMillis

  private val queue
  : LinkedBlockingQueue[BatchPoolDrawContext] = new LinkedBlockingQueue(queueLength)

  private val yieldRequest
  : AtomicBoolean = new AtomicBoolean(false)

  private val thread
  : Thread = {
    val runnable = new Runnable {
      override def run()
      : Unit = {
        var next: BatchPoolDrawContext = null
        try {
          // Draw first batch.
          next = source.draw()
          if (next == null) {
            return
          }

          // As long as yield has not been requested.
          while (!yieldRequest.get()) {
            // Keep offering this batch to the queue until it can be accepted.
            if (queue.offer(next, queueOfferTimeout, TimeUnit.MILLISECONDS)) {
              next = source.draw()
              if (next == null) {
                return
              }
            }
          }

          // Close last drawn draw context.
          next.close()
        }
        catch {
          case e: Exception =>
            logger.error(s"AdvancedPrefetcher.run: ", e)
            yieldRequest.set(true)
            throw e
        }
      }
    }

    new Thread(
      runnable,
      s"Blaze.AdvancedPrefetcher[$queueLength, $queueOfferTimeout]"
    )
  }
  thread.start()

  override protected def doClose()
  : Unit = {
    yieldRequest.set(true)
    thread.join()
    val iter = queue.iterator()
    while (iter.hasNext) {
      iter.next().close()
    }
    super.doClose()
  }

  override def draw()
  : BatchPoolDrawContext = queue.take()

}

final class AdvancedPrefetcherBuilder
  extends PrefetcherBuilder[AdvancedPrefetcherBuilder] {

  override def repr
  : AdvancedPrefetcherBuilder = this

  private var _queueLength
  : Int = 5

  def queueLength
  : Int = _queueLength

  def queueLength_=(value: Int)
  : Unit = {
    require(value > 0)
    _queueLength = value
  }

  def setQueueLength(value: Int)
  : AdvancedPrefetcherBuilder = {
    queueLength_=(value)
    this
  }

  private var _queueOfferTimeout
  : TimeSpan = TimeSpan.oneSecond

  def queueOfferTimeout
  : TimeSpan = _queueOfferTimeout

  def queueOfferTimeout_=(value: TimeSpan)
  : Unit = {
    require(value != null)
    _queueOfferTimeout = value
  }

  def setQueueOfferTimeout(value: TimeSpan)
  : AdvancedPrefetcherBuilder = {
    queueOfferTimeout_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = _queueLength :: _queueOfferTimeout :: super.doToString()

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _queueLength.hashCode())
    tmp = MurmurHash3.mix(tmp, _queueOfferTimeout.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[AdvancedPrefetcherBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: AdvancedPrefetcherBuilder =>
      _queueLength       == other._queueLength &&
      _queueOfferTimeout == other._queueOfferTimeout
    case _ =>
      false
  })

  override def doCopy()
  : AdvancedPrefetcherBuilder = AdvancedPrefetcherBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: AdvancedPrefetcherBuilder =>
        other._queueLength       = _queueLength
        other._queueOfferTimeout = _queueOfferTimeout
      case _ =>
    }
  }

  override protected def doBuild(source: BatchPool, seed: InstanceSeed)
  : BatchPool = new AdvancedPrefetcher(this, seed, source)

}

object AdvancedPrefetcherBuilder {

  final def apply()
  : AdvancedPrefetcherBuilder = new AdvancedPrefetcherBuilder

  final def apply(source: BatchPoolBuilder)
  : AdvancedPrefetcherBuilder = apply().setSource(source)

  final def apply(source: BatchPoolBuilder, queueLength: Int)
  : AdvancedPrefetcherBuilder = apply(source).setQueueLength(queueLength)

}