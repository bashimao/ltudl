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
import java.awt.Color
import scala.util.hashing._

abstract class Element
  extends Equatable
    with Serializable {

  def repr
  : Element

  def defaultLabel()
  : Option[String]

  final private var _label
  : Option[String] = defaultLabel()

  final def label
  : Option[String] = _label

  final def label_=(value: Option[String])
  : Unit = {
    require(value != null)
    _label = value
  }

  final def label_=(value: String)
  : Unit = label_=(Option(value))

  def setLabel(value: String)
  : Element

  def setLabel(value: Option[String])
  : Element

  def defaultLabelColor()
  : Color

  final private var _labelColor
  : Color = defaultLabelColor()

  final def labelColor
  : Color = _labelColor

  final def labelColor_=(value: Color)
  : Unit = {
    require(value != null)
    _labelColor = value
  }

  def setLabelColor(value: Color)
  : Element

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, _label.hashCode())
    tmp = MurmurHash3.mix(tmp, _labelColor.hashCode())
    tmp
  }

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Element =>
      _label      == other._label &&
      _labelColor == other._labelColor
    case _ =>
      false
  })

}

trait ElementEx[TThis <: ElementEx[_]]
  extends Element {

  override def repr
  : TThis

  final override def setLabel(value: String)
  : TThis = setLabel(Option(value))

  final override def setLabel(value: Option[String])
  : TThis = {
    label_=(value)
    repr
  }

  final override def setLabelColor(value: Color)
  : TThis = {
    labelColor_=(value)
    repr
  }

}
