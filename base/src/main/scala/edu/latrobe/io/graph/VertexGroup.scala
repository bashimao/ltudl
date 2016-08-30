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
import java.util.UUID
import scala.collection._
import scala.util.hashing._

final class VertexGroup(override val id: UUID)
  extends NodeEx[VertexGroup] {

  override def repr
  : VertexGroup = this

  override def defaultShape()
  : NodeShape = NodeShape.RoundedBox

  override def defaultLabel()
  : Option[String] = Some(id.toString)

  override def defaultLabelColor()
  : Color = DefaultColors.gray

  override def defaultOutlineStyle()
  : Option[LineStyle] = Some(LineStyle.Dashed)

  override def defaultOutlineColor()
  : Color = DefaultColors.gray

  override def defaultFillColor()
  : Option[Color] = Some(DefaultColors.white)

  val children
  : mutable.Buffer[Node] = mutable.Buffer.empty

  def +=(node: Node)
  : VertexGroup = {
    children += node
    this
  }

  def ++=(nodes: TraversableOnce[Node])
  : VertexGroup = {
    children ++= nodes
    this
  }

  override def toString
  : String = s"$label[${children.length}]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), children.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[VertexGroup]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: VertexGroup =>
      children == other.children
    case _ =>
      false
  })

}

object VertexGroup {

  final def apply()
  : VertexGroup = apply(UUID.randomUUID())

  final def apply(child0: Node)
  : VertexGroup = apply() += child0

  final def apply(child0: Node, childN: Node*)
  : VertexGroup = apply(child0) ++= childN

  final def apply(childN: TraversableOnce[Node])
  : VertexGroup = apply() ++= childN

  final def apply(id: UUID)
  : VertexGroup = new VertexGroup(id)

  final def apply(id: UUID, child0: Node)
  : VertexGroup = apply(id) += child0

  final def apply(id: UUID, child0: Node, childN: Node*)
  : VertexGroup = apply(id, child0) ++= childN

  final def apply(id: UUID, childN: TraversableOnce[Node])
  : VertexGroup = apply(id) ++= childN

  final def derive(label: String)
  : VertexGroup = apply().setLabel(label)

}
