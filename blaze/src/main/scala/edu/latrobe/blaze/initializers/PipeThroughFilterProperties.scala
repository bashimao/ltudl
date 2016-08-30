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

/**
  * Useful to mimic Torch initialization of bias.
  */
final class PipeThroughFilterProperties(override val builder: PipeThroughFilterPropertiesBuilder,
                                        override val seed:    InstanceSeed)
  extends DependentInitializer[PipeThroughFilterPropertiesBuilder] {

  private var inputFanSizeOfPreviousFilter
  : Int = -1

  private var outputFanSizeOfPreviousFilter
  : Int = -1

  override def apply(module:        Module,
                     reference:     LabeledBufferReference,
                     weights:       ValueTensor,
                     inputFanSize:  Int,
                     outputFanSize: Int)
  : Unit = {
    if (reference.handle == "filter") {
      inputFanSizeOfPreviousFilter  = inputFanSize
      outputFanSizeOfPreviousFilter = outputFanSize
    }
    super.apply(
      module,
      reference,
      weights,
      inputFanSizeOfPreviousFilter,
      outputFanSizeOfPreviousFilter
    )
  }

  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : PipeThroughFilterPropertiesState = PipeThroughFilterPropertiesState(
    super.state, inputFanSizeOfPreviousFilter, outputFanSizeOfPreviousFilter
  )

  override def restoreState(state: InstanceState): Unit = {
    super.restoreState(state.parent)
    state match {
      case state: PipeThroughFilterPropertiesState =>
        inputFanSizeOfPreviousFilter  = state.inputFanSizeOfPreviousFilter
        outputFanSizeOfPreviousFilter = state.outputFanSizeOfPreviousFilter
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class PipeThroughFilterPropertiesBuilder
  extends DependentInitializerBuilder[PipeThroughFilterPropertiesBuilder] {

  override def repr
  : PipeThroughFilterPropertiesBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[PipeThroughFilterPropertiesBuilder]

  override protected def doCopy()
  : PipeThroughFilterPropertiesBuilder = PipeThroughFilterPropertiesBuilder()

  override def build(seed: InstanceSeed)
  : PipeThroughFilterProperties = new PipeThroughFilterProperties(this, seed)

}

object PipeThroughFilterPropertiesBuilder {

  final def apply()
  : PipeThroughFilterPropertiesBuilder = new PipeThroughFilterPropertiesBuilder

}

final case class PipeThroughFilterPropertiesState(override val parent:           InstanceState,
                                                  inputFanSizeOfPreviousFilter:  Int,
                                                  outputFanSizeOfPreviousFilter: Int)
  extends InstanceState {
}
