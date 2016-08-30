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

package edu.latrobe.blaze

import edu.latrobe.JSerializable

abstract class Sink
  extends InstanceEx[SinkBuilder] {

  def write(src0: Any)
  : Unit

  def writeRaw(src0: Array[Byte])
  : Unit

  def writeRaw(src0: JSerializable)
  : Unit

  override def state
  : SinkState = SinkStateEx(super.state)

  /**
    * However, restoring a state is always safe. Hence, the state is merely
    * copied into the object.
    */
  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: SinkStateEx =>
      case _ =>
        throw new MatchError(state)
    }
  }

}

/**
  * A portable generic wrapper that allows us to write in multiple destinations.
  */
abstract class SinkBuilder
  extends InstanceExBuilder0[SinkBuilder, Sink] {
}

abstract class SinkEx[TBuilder <: SinkExBuilder[_]]
  extends Sink {

  def builder
  : TBuilder

}


abstract class SinkExBuilder[TBuilder <: SinkExBuilder[_]]
  extends SinkBuilder {

  def repr
  : TBuilder

  override protected def doCopy()
  : TBuilder

}

abstract class SinkState
  extends InstanceState {
}

final case class SinkStateEx(override val parent: InstanceState)
  extends SinkState {
}
