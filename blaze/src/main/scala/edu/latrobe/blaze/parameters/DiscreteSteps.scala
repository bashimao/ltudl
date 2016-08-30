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

package edu.latrobe.blaze.parameters

import edu.latrobe._
import edu.latrobe.blaze._
import scala.collection._
import scala.util.hashing._

final class DiscreteSteps(override val builder: DiscreteStepsBuilder,
                          override val name:    String,
                          override val seed:    InstanceSeed)
  extends IndependentParameter[DiscreteStepsBuilder] {

  private val steps
  : Array[(Long, Real)] = builder.steps.toArray
  assume(steps.length > 0)

  override def get(phaseNo: Long)
  : Real = {
    var i = steps.length - 1
    while (i > 0) {
      val step = steps(i)
      if (phaseNo >= step._1) {
        return step._2
      }
      i -= 1
    }
    steps(0)._2
  }

  override def update(phaseNo: Long, value:  Real)
  : Unit = {}

}

final class DiscreteStepsBuilder
  extends IndependentParameterBuilder[DiscreteStepsBuilder] {

  override def repr
  : DiscreteStepsBuilder = this

  val steps
  : mutable.Buffer[(Long, Real)] = mutable.Buffer.empty

  def +=(step: (Long, Real))
  : DiscreteStepsBuilder = {
    steps += step
    this
  }

  def ++=(steps: TraversableOnce[(Long, Real)])
  : DiscreteStepsBuilder = {
    this.steps ++= steps
    this
  }

  override protected def doToString()
  : List[Any] = steps.size :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), steps.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[DiscreteStepsBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: DiscreteStepsBuilder =>
      steps == other.steps
    case _ =>
      false
  })

  override protected def doCopy()
  : DiscreteStepsBuilder = DiscreteStepsBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: DiscreteStepsBuilder =>
        other.steps.clear()
        other.steps ++= steps
      case _ =>
    }
  }

  override def build(name: String, seed: InstanceSeed)
  : DiscreteSteps = new DiscreteSteps(this, name, seed)

}

object DiscreteStepsBuilder {

  final def apply()
  : DiscreteStepsBuilder = new DiscreteStepsBuilder

  final def apply(step0: (Long, Real))
  : DiscreteStepsBuilder = apply() += step0

  final def apply(step0: (Long, Real), steps: (Long, Real)*)
  : DiscreteStepsBuilder = apply(step0) ++= steps

  final def apply(steps: Seq[(Long, Real)])
  : DiscreteStepsBuilder = apply() ++= steps

}

final case class DiscreteStepsState(override val parent: InstanceState,
                                    stepNo:              Long)
  extends InstanceState {
}
