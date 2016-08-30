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

final class PrintIterationNo(override val builder: PrintIterationNoBuilder,
                             override val seed:    InstanceSeed)
  extends Print[PrintIterationNoBuilder] {

  override protected def doEvaluate(optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : String = optimizer.iterationNo.toString

}

final class PrintIterationNoBuilder
  extends PrintBuilder[PrintIterationNoBuilder] {

  override def repr
  : PrintIterationNoBuilder = ???

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[PrintIterationNoBuilder]

  override protected def doCopy()
  : PrintIterationNoBuilder = PrintIterationNoBuilder()

  override def build(seed: InstanceSeed)
  : PrintIterationNo = new PrintIterationNo(this, seed)

}

object PrintIterationNoBuilder {

  final def apply()
  : PrintIterationNoBuilder = new PrintIterationNoBuilder

}
