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

package edu.latrobe.blaze.objectives.visual

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.time._

abstract class XAxisType
  extends Serializable {

  def label
  : String

  def valueFor(optimizer: OptimizerLike)
  : Real

}

object XAxisType {

  case object IterationNo
    extends XAxisType {

    override def label
    : String = "Iteration#"

    override def valueFor(optimizer: OptimizerLike)
    : Real = Real(optimizer.iterationNo)

  }

  case object TimeInSeconds
    extends XAxisType {

    override def label
    : String = "Time (s)"

    override def valueFor(optimizer: OptimizerLike)
    : Real = {
      val now = Timestamp.now()
      TimeSpan(optimizer.beginTime, now).seconds
    }

  }

}
