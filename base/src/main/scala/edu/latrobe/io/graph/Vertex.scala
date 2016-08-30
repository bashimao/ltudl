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

import edu.latrobe.io._
import java.awt.Color
import java.util._

final class Vertex(val id: UUID)
  extends NodeEx[Vertex] {

  override def repr
  : Vertex = this

  override def defaultShape()
  : NodeShape = NodeShape.RoundedBox

  override def defaultLabel()
  : Option[String] = Some(id.toString)

  override def defaultLabelColor()
  : Color = DefaultColors.black

  override def defaultOutlineStyle()
  : Option[LineStyle] = Some(LineStyle.Solid)

  override def defaultOutlineColor()
  : Color = DefaultColors.black

  override def defaultFillColor()
  : Option[Color] = Some(DefaultColors.lightGray)

  override def toString
  : String = s"[$label]"

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Vertex]

}

object Vertex {

  final def apply()
  : Vertex = apply(UUID.randomUUID())

  final def apply(id: UUID)
  : Vertex = new Vertex(id)

  final def derive(label: String)
  : Vertex = apply().setLabel(label)

}
