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

import java.awt.Color
import java.util.UUID

import edu.latrobe.Equatable

import scala.util.hashing.MurmurHash3

abstract class Node
  extends Element {

  def repr
  : Node

  /**
    * Must override with constructor argument.
    */
  def id
  : UUID

  def defaultShape()
  : NodeShape

  final private var _shape
  : NodeShape = defaultShape()

  final def shape
  : NodeShape = _shape

  final def shape_=(value: NodeShape)
  : Unit = {
    require(value != null)
    _shape = value
  }

  def setShape(value: NodeShape)
  : Node

  def defaultOutlineStyle()
  : Option[LineStyle]

  final private var _outlineStyle
  : Option[LineStyle] = defaultOutlineStyle()

  final def outlineStyle
  : Option[LineStyle] = _outlineStyle

  final def outlineStyle_=(value: Option[LineStyle])
  : Unit = {
    require(value != null)
    _outlineStyle = value
  }

  def setOutlineStyle(value: Option[LineStyle])
  : Node

  def setOutlineStyle(value: LineStyle)
  : Node

  def defaultOutlineColor()
  : Color

  final private var _outlineColor
  : Color = defaultOutlineColor()

  final def outlineColor
  : Color = _outlineColor

  final def outlineColor_=(value: Color)
  : Unit = {
    require(value != null)
    _outlineColor = value
  }

  def setOutlineColor(value: Color)
  : Node

  def defaultFillColor()
  : Option[Color]

  final private var _fillColor
  : Option[Color] = defaultFillColor()

  final def fillColor
  : Option[Color] = _fillColor

  final def fillColor_=(value: Option[Color])
  : Unit = {
    require(value != null)
    _fillColor = value
  }

  def setFillColor(value: Color)
  : Node

  def setFillColor(value: Option[Color])
  : Node

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _shape.hashCode())
    tmp = MurmurHash3.mix(tmp, _outlineStyle.hashCode())
    tmp = MurmurHash3.mix(tmp, _outlineColor.hashCode())
    tmp = MurmurHash3.mix(tmp, _fillColor.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Node =>
      _shape        == other._shape        &&
      _outlineStyle == other._outlineStyle &&
      _outlineColor == other._outlineColor &&
      _fillColor    == other._fillColor
    case _ =>
      false
  })

}

abstract class NodeEx[TThis <: NodeEx[_]]
  extends Node
    with ElementEx[TThis] {

  def repr
  : TThis

  final override def setShape(value: NodeShape)
  : TThis = {
    shape_=(value)
    repr
  }

  final override def setOutlineStyle(value: LineStyle)
  : TThis = setOutlineStyle(Option(value))

  final override def setOutlineStyle(value: Option[LineStyle])
  : TThis = {
    outlineStyle_=(value)
    repr
  }

  final override def setOutlineColor(value: Color)
  : TThis = {
    outlineColor_=(value)
    repr
  }

  final override def setFillColor(value: Color)
  : TThis = setFillColor(Option(value))

  final override def setFillColor(value: Option[Color])
  : TThis = {
    fillColor_=(value)
    repr
  }

}

