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

package edu.latrobe.io.graph

import edu.latrobe._
import edu.latrobe.io._
import java.awt.Color

import scala.util.hashing.MurmurHash3

final class Edge(val start: Vertex,
                 val end:   Vertex)
  extends Element
    with ElementEx[Edge] {

  override def repr
  : Edge = this

  override def defaultLabel()
  : Option[String] = None

  override def defaultLabelColor()
  : Color = DefaultColors.black

  private var _style
  : Option[LineStyle] = Some(LineStyle.Solid)

  def style
  : Option[LineStyle] = _style

  def style_=(value: Option[LineStyle])
  : Unit = {
    require(value != null)
    _style = value
  }

  def setStyle(value: LineStyle)
  : Edge = setStyle(Option(value))

  def setStyle(value: Option[LineStyle])
  : Edge = {
    style_=(value)
    this
  }

  private var _headShape
  : Option[ArrowHeadShape] = Some(ArrowHeadShape.V)

  def headShape
  : Option[ArrowHeadShape] = _headShape

  def headShape_=(value: Option[ArrowHeadShape])
  : Unit = {
    require(value != null)
    _headShape = value
  }

  def setHeadShape(value: ArrowHeadShape)
  : Edge = setHeadShape(Option(value))

  def setHeadShape(value: Option[ArrowHeadShape])
  : Edge = {
    headShape_=(value)
    this
  }

  private var _tailShape
  : Option[ArrowHeadShape] = None

  def tailShape
  : Option[ArrowHeadShape] = _tailShape

  def tailShape_=(value: Option[ArrowHeadShape])
  : Unit = {
    require(value != null)
    _tailShape = value
  }

  def setTailShape(value: ArrowHeadShape)
  : Edge = setTailShape(Option(value))

  def setTailShape(value: Option[ArrowHeadShape])
  : Edge = {
    tailShape_=(value)
    this
  }

  // TODO: Add these options!
  /*
  var headLabel
  : Option[String] = None

  var headLabelColor
  : Color = Color.BLACK

  var tailLabel
  : Option[String] = None

  var tailLabelColor
  : Color = Color.BLACK
  */

  override def toString
  : String = s"$start -> $end"

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, start.hashCode())
    tmp = MurmurHash3.mix(tmp, end.hashCode())
    tmp = MurmurHash3.mix(tmp, _style.hashCode())
    tmp = MurmurHash3.mix(tmp, _headShape.hashCode())
    tmp = MurmurHash3.mix(tmp, _tailShape.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Edge]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Edge =>
      start      == other.start      &&
      end        == other.end        &&
      _style     == other._style     &&
      _headShape == other._headShape &&
      _tailShape == other._tailShape
    case _ =>
      false
  })

}

object Edge {

  final def apply(start: Vertex,
                  end:   Vertex)
  : Edge = new Edge(start, end)

  final def apply(start: Vertex,
                  end:   Vertex,
                  style: LineStyle)
  : Edge = apply(start, end).setStyle(style)

}
