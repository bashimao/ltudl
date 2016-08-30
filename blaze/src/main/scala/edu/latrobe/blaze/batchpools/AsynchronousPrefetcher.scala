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
import scala.concurrent._
import scala.concurrent.ExecutionContext.Implicits.global

/**
  * This fancy block is a proxy that uses futures to request the next batch
  * ahead of time.
  */
final class AsynchronousPrefetcher(override val builder: AsynchronousPrefetcherBuilder,
                                   override val seed:    InstanceSeed,
                                   override val source:  BatchPool)
  extends Prefetcher[AsynchronousPrefetcherBuilder] {

  override val outputHints
  : BuildHints = inputHints

  private var next
  : Future[BatchPoolDrawContext] = _

  override protected def doClose()
  : Unit = {
    if (next != null) {
      val src = FutureEx.get(next)
      src.close()
      next = null
    }
    super.doClose()
  }

  override def draw()
  : BatchPoolDrawContext = {
    // If next not initialized.
    if (next == null) {
      next = Future.successful(source.draw())
    }

    // Wait for the next batch to arrive.
    val src = FutureEx.get(next)

    // Queue the next batch.
    next = Future({
      //blazeLogger.trace("Prefeching batch data...")
      //val begin = Timestamp()
      val src = source.draw()
      //val end   = Timestamp()
      //blazeLogger.trace(f"Prefeching batch data complete! (${TimeSpan(begin, end).seconds}%.3f s)")
      src
    })

    // Return the src.
    src
  }

}

final class AsynchronousPrefetcherBuilder
  extends PrefetcherBuilder[AsynchronousPrefetcherBuilder] {

  override def repr: AsynchronousPrefetcherBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[AsynchronousPrefetcherBuilder]

  override def doCopy()
  : AsynchronousPrefetcherBuilder = AsynchronousPrefetcherBuilder()


  // ---------------------------------------------------------------------------
  //   Record set construction
  // ---------------------------------------------------------------------------
  override protected def doBuild(source: BatchPool,
                                 seed:   InstanceSeed)
  : AsynchronousPrefetcher = new AsynchronousPrefetcher(this, seed, source)

}

object AsynchronousPrefetcherBuilder {

  final def apply()
  : AsynchronousPrefetcherBuilder = new AsynchronousPrefetcherBuilder

  final def apply(source: BatchPoolBuilder)
  : AsynchronousPrefetcherBuilder = apply().setSource(source)

}
