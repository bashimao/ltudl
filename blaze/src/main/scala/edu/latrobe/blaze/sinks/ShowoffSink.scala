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

package edu.latrobe.blaze.sinks

import edu.latrobe._
import edu.latrobe.io.showoff._
import edu.latrobe.blaze._
import edu.latrobe.time._
import scala.util.hashing._

/**
  * Redirects output to showoff frame. Certain frame formats only support
  * certain data. You may want to catch all sorts of exceptions since this
  * travels over the TCP/IP stack.
  */
final class ShowoffSink(override val builder: ShowoffSinkBuilder,
                        override val seed:    InstanceSeed)
  extends SinkEx[ShowoffSinkBuilder] {

  val frameHandle
  : String = builder.frameHandle

  val frameTitle
  : String = builder.frameTitle

  val frameFormat
  : String = builder.frameFormat

  val renderInterval
  : TimeSpan = builder.renderInterval

  val notebook
  : Notebook = Notebook.get(seed.agentNo)

  val frame
  : Frame = notebook.getOrCreateFrame(frameHandle)

  private var clock
  : Timer = Timer(renderInterval)

  override def write(src0: Any)
  : Unit = {
    if (clock.resetIfElapsed(renderInterval)) {
      frame.render(frameHandle, frameFormat, src0)
    }
  }

  override def writeRaw(src0: Array[Byte])
  : Unit = throw new UnsupportedOperationException

  override def writeRaw(src0: JSerializable)
  : Unit = throw new UnsupportedOperationException

  override def state
  : SinkState = ShowoffSinkState(super.state, clock)

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state)
    state match {
      case state: ShowoffSinkState =>
        clock = state.clock.copy
      case _ =>
        throw new MatchError(state)
    }
  }

}


final class ShowoffSinkBuilder
  extends SinkExBuilder[ShowoffSinkBuilder] {

  override def repr
  : ShowoffSinkBuilder = this

  private var _frameHandle
  : String = id.toString

  def frameHandle
  : String = _frameHandle

  def frameHandle_=(value: String)
  : Unit = {
    require(value != null)
    _frameHandle = value
  }

  def setFrameHandle(value: String)
  : ShowoffSinkBuilder = {
    frameHandle_=(value)
    this
  }

  private var _frameTitle
  : String = s"edu.latrobe.blaze.sinks.ShowoffSink - ${_frameHandle}"

  def frameTitle
  : String = _frameTitle

  def frameTitle_=(value: String)
  : Unit = {
    require(value != null)
    _frameTitle = value
  }
  def setFrameTitle(value: String)
  : ShowoffSinkBuilder = {
    frameTitle_=(value)
    repr
  }

  private var _frameFormat
  : String = "best"

  def frameFormat
  : String = _frameFormat

  def frameFormat_=(value: String)
  : Unit = {
    require(value != null)
    _frameFormat = value
  }

  def setFrameFormat(value: String)
  : ShowoffSinkBuilder = {
    frameFormat_=(value)
    this
  }

  private var _renderInterval
  : TimeSpan = TimeSpan.zero

  def renderInterval
  : TimeSpan = _renderInterval

  def renderInterval_=(value: TimeSpan)
  : Unit = {
    require(value != null)
    _renderInterval = value
  }

  def setRenderInterval(value: TimeSpan)
  : ShowoffSinkBuilder = {
    renderInterval_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = {
    _frameHandle :: _frameTitle :: _frameFormat :: _renderInterval :: super.doToString()
  }

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _frameHandle.hashCode())
    tmp = MurmurHash3.mix(tmp, _frameTitle.hashCode())
    tmp = MurmurHash3.mix(tmp, _frameFormat.hashCode())
    tmp = MurmurHash3.mix(tmp, _renderInterval.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ShowoffSinkBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: ShowoffSinkBuilder =>
      _frameHandle    == other._frameHandle &&
      _frameTitle     == other._frameTitle  &&
      _frameFormat    == other._frameFormat &&
      _renderInterval == other._renderInterval
    case _ =>
      false
  })

  override protected def doCopy()
  : ShowoffSinkBuilder = ShowoffSinkBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: ShowoffSinkBuilder =>
        other._frameHandle    = _frameHandle
        other._frameTitle     = _frameTitle
        other._frameFormat    = _frameFormat
        other._renderInterval = _renderInterval
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : ShowoffSink = new ShowoffSink(this, seed)

}

object ShowoffSinkBuilder {

  final def apply()
  : ShowoffSinkBuilder = new ShowoffSinkBuilder

  final def apply(frameTitle: String)
  : ShowoffSinkBuilder = apply().setFrameTitle(
    frameTitle
  ).setFrameHandle(frameTitle)

  final def apply(frameTitle:     String,
                  renderInterval: TimeSpan)
  : ShowoffSinkBuilder = apply(
    frameTitle
  ).setRenderInterval(renderInterval)

}

final case class ShowoffSinkState(override val parent: InstanceState,
                                  clock:               Timer)
  extends SinkState {
}
