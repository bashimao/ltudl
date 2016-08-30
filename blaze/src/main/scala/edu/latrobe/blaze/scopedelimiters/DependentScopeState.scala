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

package edu.latrobe.blaze.scopedelimiters

import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

abstract class DependentScope[TBuilder <: DependentScopeBuilder[_]]
  extends ScopeDelimiterEx[TBuilder] {

  /**
    * Must be implemented as constructor argument!
    */
  def source
  : ScopeDelimiter


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : InstanceState = DependentScopeState(super.state, source.state)

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: DependentScopeState =>
        source.restoreState(state.source)
      case _ =>
        throw new MatchError(state)
    }
  }

}

abstract class DependentScopeBuilder[TThis <: DependentScopeBuilder[_]]
  extends ScopeDelimiterExBuilder[TThis] {

  final private var _source
  : ScopeDelimiterBuilder = EntireBufferBuilder()

  final def source
  : ScopeDelimiterBuilder = _source

  final def source_=(value: ScopeDelimiterBuilder)
  : Unit = {
    require(value != null)
    _source = value
  }

  final def setSource(value: ScopeDelimiterBuilder)
  : TThis = {
    source_=(value)
    repr
  }

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _source.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: DependentScopeBuilder[_] =>
      _source == other._source
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: DependentScopeBuilder[_] =>
        other._source = _source
      case _ =>
    }
  }

  final override def build(source: NullBuffer,
                           seed:   InstanceSeed)
  : DependentScope[TThis] = doBuild(_source.build(source, seed))

  protected def doBuild(source: ScopeDelimiter)
  : DependentScope[TThis]

}

final case class DependentScopeState(override val parent: InstanceState,
                                     source:              InstanceState)
  extends InstanceState {
}
