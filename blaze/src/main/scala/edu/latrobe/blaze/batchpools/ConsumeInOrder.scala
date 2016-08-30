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
import scala.collection._

/**
 * The probably most simple variant of a record series. Just read everything
 * from front to back. No overhead.
 */
final class ConsumeInOrder(override val builder:    ConsumeInOrderBuilder,
                           override val inputHints: BuildHints,
                           override val seed:       InstanceSeed,
                           var          samples:    Iterator[Batch])
  extends IndependentBatchPool[ConsumeInOrderBuilder] {

  private var sampleNo
  : Long = 0L

  override def draw()
  : IndependentBatchPoolDrawContext = {
    if (samples.hasNext) {
      IndependentBatchPoolDrawContext(samples.next())
    }
    else {
      IndependentBatchPoolDrawContext(null)
    }
  }


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : BatchPoolState = ConsumeInOrderState(super.state, sampleNo)

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: ConsumeInOrderState =>
        while (sampleNo != state.sampleNo) {
          samples.next()
          sampleNo += 1L
        }
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class ConsumeInOrderBuilder
  extends IndependentBatchPoolBuilder[ConsumeInOrderBuilder] {

  override def repr
  : ConsumeInOrderBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ConsumeInOrderBuilder]

  override protected def doCopy()
  : ConsumeInOrderBuilder = ConsumeInOrderBuilder()


  // ---------------------------------------------------------------------------
  //   Record set construction
  // ---------------------------------------------------------------------------
  /**
    * Recursively build all pools, and feeds the provided iterator to the root
    * pool.
    */
  override def build(layoutHint: TensorLayout,
                     samples:    Iterable[Batch],
                     seed:       InstanceSeed)
  : ConsumeInOrder = new ConsumeInOrder(
    this,
    BuildHints(JVM, layoutHint),
    seed,
    samples.toIterator
  )

}

object ConsumeInOrderBuilder {

  final def apply()
  : ConsumeInOrderBuilder = new ConsumeInOrderBuilder

}

final case class ConsumeInOrderState(override val parent: InstanceState,
                                     sampleNo:            Long)
  extends BatchPoolState
