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

import edu.latrobe.blaze._
import scala.collection._

/**
  * A super simple container that adds no functionality for structuring
  * purposes.
  */
final class MultiObjective(override val builder: MultiObjectiveBuilder,
                           override val seed:    InstanceSeed)
  extends DependentObjectiveEx[MultiObjectiveBuilder] {
  require(builder != null && seed != null)
}

final class MultiObjectiveBuilder
  extends DependentObjectiveExBuilder[MultiObjectiveBuilder] {

  override def repr
  : MultiObjectiveBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[MultiObjectiveBuilder]

  override protected def doCopy()
  : MultiObjectiveBuilder = MultiObjectiveBuilder()

  override def build(seed: InstanceSeed)
  : MultiObjective = new MultiObjective(this, seed)

}

object MultiObjectiveBuilder {

  final def apply()
  : MultiObjectiveBuilder = new MultiObjectiveBuilder

  final def apply(child0: ObjectiveBuilder)
  : MultiObjectiveBuilder = apply() += child0

  final def apply(child0: ObjectiveBuilder, childN: ObjectiveBuilder*)
  : MultiObjectiveBuilder = apply(child0) ++= childN

  final def apply(childN: TraversableOnce[ObjectiveBuilder])
  : MultiObjectiveBuilder = apply() ++= childN

  final def apply(childN: Array[ObjectiveBuilder])
  : MultiObjectiveBuilder = apply() ++= childN

}
