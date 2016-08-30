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
import edu.latrobe.blaze._
import edu.latrobe.time._

/**
  * Merely transforms the time into a value and inserts it as such into child
  * modules.
  */
final class SelectTimeInSeconds(override val builder: SelectTimeInSecondsBuilder,
                                override val seed:    InstanceSeed)
  extends ReplaceValue[SelectTimeInSecondsBuilder] {

  override protected def doEvaluate(optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : Real = {
    val result = TimeSpan(optimizer.beginTime, Timestamp.now())
    result.seconds
  }

}

final class SelectTimeInSecondsBuilder
  extends ReplaceValueBuilder[SelectTimeInSecondsBuilder] {

  override def repr
  : SelectTimeInSecondsBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[SelectTimeInSecondsBuilder]

  override protected def doCopy()
  : SelectTimeInSecondsBuilder = SelectTimeInSecondsBuilder()

  override def build(seed: InstanceSeed)
  : SelectTimeInSeconds = new SelectTimeInSeconds(this, seed)

}

object SelectTimeInSecondsBuilder {

  final def apply()
  : SelectTimeInSecondsBuilder = new SelectTimeInSecondsBuilder

}
