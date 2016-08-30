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

package edu.latrobe.io.graph.renderers

import edu.latrobe.io.graph._
import java.awt.Color
import java.io.{OutputStreamWriter, OutputStream, Writer}
import java.util.UUID
import scala.collection._

object DotRenderer
  extends GraphRenderer {

  override def render(graph: Graph, stream: OutputStream)
  : Unit = render(graph, new OutputStreamWriter(stream))

  final override def render(graph: Graph, writer: Writer)
  : Unit = {
    def write(values: String*)
    : Unit = values.foreach(writer.write)

    def writeLine(values: String*)
    : Unit = {
      values.foreach(writer.write)
      writer.write(System.lineSeparator())
    }

    writeLine("digraph G {")

    // Render nodes.
    render(graph.nodes, writer)

    // Render edges.
    graph.edges.foreach(edge => {
      write(render(edge.start.id), " -> ", render(edge.end.id))
      write(" [")
      write("dir=", "both")
      write(", arrowhead=", renderArrowHeadShape(edge.headShape))
      write(", arrowtail=", renderArrowHeadShape(edge.tailShape))
      write(", style=", renderLineStyle(edge.style))
      for (label <- edge.label) {
        write(", label=", render(" " + label.replace("\n", " \\n")))
        write(", fontsize=", "8")
        write(", fontcolor=", render(edge.labelColor))
        //write(", labelfloat=", "true")
      }
      writeLine("]")
    })

    writeLine("}")
    writer.flush()
  }

  @inline
  final private  def render(nodes:  Traversable[Node],
                            writer: Writer)
  : Unit = {
    def write(values: String*)
    : Unit = values.foreach(writer.write)

    def writeLine(values: String*)
    : Unit = {
      values.foreach(writer.write)
      writer.write(System.lineSeparator())
    }

    nodes.foreach(node => {
      node match {
        case node: VertexGroup =>
          write("subgraph")
          writeLine(render("cluster" + node.id.toString))
          writeLine("{")

          // Outline
          val style = StringBuilder.newBuilder
          style ++= renderLineStyle(node.outlineStyle)
          writeLine("color=", render(node.outlineColor))

          // Shape
          writeLine("shape=", render(node.shape))
          node.shape match {
            case NodeShape.RoundedBox =>
              style ++= ", rounded"
            case _ =>
          }

          // Label
          node.label.foreach(label => {
            writeLine("label=", render(label))
          })
          writeLine("fontcolor=", render(node.labelColor))

          // Fill
          node.fillColor.foreach(color => {
            writeLine("fillcolor=", render(color))
            style ++= ", filled"
          })

          // Style
          if (style.nonEmpty) {
            writeLine("style=", render(style.result()))
          }

          // Children
          render(node.children, writer)

          writeLine("}")

        case node: Vertex =>
          write(render(node.id.toString))
          write(" [")

          // Outline
          val style = StringBuilder.newBuilder
          style ++= renderLineStyle(node.outlineStyle)
          write("color=", render(node.outlineColor))

          // Shape
          write(", shape=", render(node.shape))
          node.shape match {
            case NodeShape.RoundedBox =>
              style ++= ", rounded"
            case _ =>
          }

          // Label
          node.label.foreach(label => {
            write(", label=", render(label))
          })
          write(", fontcolor=", render(node.labelColor))

          // Fill
          node.fillColor.foreach(fillColor => {
            write(", fillcolor=", render(fillColor))
            style ++= ", filled"
          })

          // Style
          if (style.nonEmpty) {
            write(", style=", render(style.result()))
          }
          writeLine("]")

        case _ =>
          throw new MatchError(node)
      }
      writeLine()
    })
  }

  @inline
  final private  def render(str: String)
  : String = s"${'"'}${str.replaceAll("\"", "\\\"")}${'"'}"

  @inline
  final private  def render(id: UUID)
  : String = render(id.toString)

  @inline
  final private def render(color: Color)
  : String = render(
    f"#${color.getRed}%02x${color.getGreen}%02x${color.getBlue}%02x"
  )

  @inline
  final private def render(shape: NodeShape)
  : String = shape match {
    case NodeShape.Box =>
      "box"
    case NodeShape.Circle =>
      "circle"
    case NodeShape.Diamond =>
      "diamond"
    case NodeShape.Ellipse =>
      "ellipse"
    case NodeShape.Hexagon =>
      "hexagon"
    case NodeShape.Octagon =>
      "octagon"
    case NodeShape.Parallelogram =>
      "parallelogram"
    case NodeShape.Point =>
      "point"
    case NodeShape.RoundedBox =>
      "box"
    case NodeShape.Triangle =>
      "triangle"
  }

  // TODO: Get somebody to fix HTML label rendering for rendering of generics.
  @inline
  final private def renderHtml(str: String)
  : String = s"${'<'}$str${'>'}"

  @inline
  final private def renderLineStyle(style: Option[LineStyle])
  : String = style match {
    case None =>
      "invis"
    case Some(LineStyle.Dashed) =>
      "dashed"
    case Some(LineStyle.Dotted) =>
      "dotted"
    case Some(LineStyle.Solid) =>
      "solid"
  }

  @inline
  final private def renderArrowHeadShape(headShape: Option[ArrowHeadShape])
  : String = headShape match {
    case None =>
      "none"
    case Some(ArrowHeadShape.Box) =>
      "box"
    case Some(ArrowHeadShape.Diamond) =>
      "diamond"
    case Some(ArrowHeadShape.Dot) =>
      "dot"
    case Some(ArrowHeadShape.InvertedTriangle) =>
      "inv"
    case Some(ArrowHeadShape.Triangle) =>
      "normal"
    case Some(ArrowHeadShape.V) =>
      "vee"
  }

}
