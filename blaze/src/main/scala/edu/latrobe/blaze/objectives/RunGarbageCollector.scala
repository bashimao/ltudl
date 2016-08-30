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

package edu.latrobe.blaze.objectives

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.time._
import java.io.OutputStream

/**
 * A pseudo target that does nothing except invoking the garbage collector.
 * (Use this only for debugging purposes!)
 */
final class RunGarbageCollector(override val builder: RunGarbageCollectorBuilder,
                                override val seed:    InstanceSeed)
  extends IndependentObjective[RunGarbageCollectorBuilder] {

  override protected def doEvaluate(sink:                Sink,
                                    optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : Option[ObjectiveEvaluationResult] = {
    System.gc()
    System.runFinalization()
    None
  }

}


final class RunGarbageCollectorBuilder
  extends IndependentObjectiveBuilder[RunGarbageCollectorBuilder] {

  override def repr
  : RunGarbageCollectorBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[RunGarbageCollectorBuilder]

  override protected def doCopy()
  : RunGarbageCollectorBuilder = RunGarbageCollectorBuilder()

  override def build(seed: InstanceSeed)
  : RunGarbageCollector = new RunGarbageCollector(this, seed)

}

object RunGarbageCollectorBuilder {

  final def apply()
  : RunGarbageCollectorBuilder = new RunGarbageCollectorBuilder

}