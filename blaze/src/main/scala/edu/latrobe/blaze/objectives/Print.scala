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

package edu.latrobe.blaze.objectives

import edu.latrobe._
import edu.latrobe.blaze.{objectives, _}
import edu.latrobe.time._
import java.io.{OutputStream, OutputStreamWriter, PrintStream}

import breeze.io.TextWriter.PrintStreamWriter

/**
  * All print objectives are essentially dummy objectives that just output stuff
  * to the currently selected stream.
  */
abstract class Print[TBuilder <: PrintBuilder[_]]
  extends ReportingObjective[TBuilder] {

  final override protected def doEvaluateEx(sink:                Sink,
                                            optimizer:           OptimizerLike,
                                            runBeginIterationNo: Long,
                                            runBeginTime:        Timestamp,
                                            runNoSamples:        Long,
                                            model:               Module,
                                            batch:               Batch,
                                            output:              Tensor,
                                            value:               Real)
  : Unit = {
    val text = doEvaluate(
      optimizer, runBeginIterationNo, runBeginTime, runNoSamples,
      model,
      batch, output, value
    )
    sink.write(text)
  }

  protected def doEvaluate(optimizer:           OptimizerLike,
                           runBeginIterationNo: Long,
                           runBeginTime:        Timestamp,
                           runNoSamples:        Long,
                           model:               Module,
                           batch:               Batch,
                           output:              Tensor,
                           value:               Real)
  : String

}

abstract class PrintBuilder[TThis <: PrintBuilder[_]]
  extends ReportingObjectiveBuilder[TThis] {
}
