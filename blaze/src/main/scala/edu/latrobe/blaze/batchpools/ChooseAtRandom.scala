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

final class ChooseAtRandom(override val builder:    ChooseAtRandomBuilder,
                           override val inputHints: BuildHints,
                           override val seed:       InstanceSeed,
                           val          batches:    Array[Batch])
  extends IndependentBatchPool[ChooseAtRandomBuilder] {

  override def draw()
  : IndependentBatchPoolDrawContext = {
    val batch = rng.next(batches)
    IndependentBatchPoolDrawContext(batch)
  }

}

final class ChooseAtRandomBuilder
  extends IndependentBatchPoolBuilder[ChooseAtRandomBuilder] {

  override def repr
  : ChooseAtRandomBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ChooseAtRandomBuilder]

  override protected def doCopy()
  : ChooseAtRandomBuilder = ChooseAtRandomBuilder()


  // ---------------------------------------------------------------------------
  //   Record set construction
  // ---------------------------------------------------------------------------
  override def build(layoutHint: TensorLayout,
                     samples:    Iterable[Batch],
                     seed:       InstanceSeed)
  : ChooseAtRandom = new ChooseAtRandom(
    this,
    BuildHints(JVM, layoutHint),
    seed,
    samples.toArray
  )

}

object ChooseAtRandomBuilder {

  final def apply()
  : ChooseAtRandomBuilder = new ChooseAtRandomBuilder

}
