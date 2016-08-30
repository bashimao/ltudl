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

package edu.latrobe.blaze.initializers

import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

/**
  * Complex initializers that operate on the outputs of other initializers. This
  * is usually used for calibration algorithms.
  */
abstract class DependentInitializer[TBuilder <: DependentInitializerBuilder[_]]
  extends InitializerEx[TBuilder] {

  final val source
  : Initializer = builder.source.build(seed)

  override def apply(module:        Module,
                     reference:     LabeledBufferReference,
                     weights:       ValueTensor,
                     inputFanSize:  Int,
                     outputFanSize: Int)
  : Unit = source.apply(
    module,
    reference,
    weights,
    inputFanSize,
    outputFanSize
  )


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state: InstanceState = DependentInitializerState(
    super.state, source.state
  )

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: DependentInitializerState =>
        source.restoreState(state.baseInitializer)
      case _ =>
        throw new MatchError(state)
    }
  }

}

abstract class DependentInitializerBuilder[TThis <: DependentInitializerBuilder[_]]
  extends InitializerExBuilder[TThis] {

  final private var _source
  : InitializerBuilder = GaussianDistributionBuilder()

  final def source
  : InitializerBuilder = _source

  final def source_=(value: InitializerBuilder): Unit = {
    require(value != null)
    _source = value
  }

  final def setSource(value: InitializerBuilder): TThis = {
    source_=(value)
    repr
  }

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _source.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: DependentInitializerBuilder[TThis] =>
      _source == other._source
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder): Unit = {
    super.copyTo(other)
    other match {
      case other: DependentInitializerBuilder[TThis] =>
        other._source = _source
      case _ =>
    }
  }

  override protected def doPermuteSeeds(fn: BuilderSeed => BuilderSeed)
  : Unit = {
    super.doPermuteSeeds(fn)
    _source.permuteSeeds(fn)
  }

}

final case class DependentInitializerState(override val parent: InstanceState,
                                           baseInitializer:     InstanceState)
  extends InstanceState {
}
