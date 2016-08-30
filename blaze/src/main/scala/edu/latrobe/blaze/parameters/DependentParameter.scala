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

package edu.latrobe.blaze.parameters

import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

abstract class DependentParameter[TBuilder <: DependentParameterBuilder[_]]
  extends ParameterEx[TBuilder] {

  final val source
  : Parameter = builder.source.build(name, seed)

  override def get(phaseNo: Long)
  : Real = source.get(phaseNo)

  override def update(phaseNo: Long, value: Real)
  : Unit = source.update(phaseNo, value)


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : InstanceState = DependentParameterState(
    super.state,
    source.state
  )

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: DependentParameterState =>
        source.restoreState(state.source)
      case _ =>
        throw new MatchError(state)
    }
  }

}

abstract class DependentParameterBuilder[TThis <: DependentParameterBuilder[_]]
  extends ParameterExBuilder[TThis] {

  final private var _source
  : ParameterBuilder = ConstantValueBuilder.nan

  final def source
  : ParameterBuilder = _source

  final def source_=(value: ParameterBuilder)
  : Unit = {
    require(value != null)
    _source = value
  }

  final def setSource(value: ParameterBuilder)
  : TThis = {
    source_=(value)
    repr
  }

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _source.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: DependentParameterBuilder[TThis] =>
      _source == other._source
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: DependentParameterBuilder[TThis] =>
        other._source = _source.copy
      case _ =>
    }
  }

  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    _source.permuteSeeds(fn)
  }

}

final case class DependentParameterState(override val parent: InstanceState,
                                         source:              InstanceState)
  extends InstanceState {
}
