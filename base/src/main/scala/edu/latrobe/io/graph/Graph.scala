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

import scala.collection._

final class Graph {

  val nodes
  : mutable.Buffer[Node] = mutable.Buffer.empty

  def +=(node: Node)
  : Graph = {
    nodes += node
    this
  }

  val edges
  : mutable.Buffer[Edge] = mutable.Buffer.empty

  def +=(edge: Edge)
  : Graph = {
    this.edges += edge
    this
  }

}

object Graph {

  final def apply()
  : Graph = new Graph

  final def apply(nodes: TraversableOnce[Node],
                  edges: TraversableOnce[Edge])
  : Graph = {
    val result = apply()
    nodes.foreach(
      result += _
    )
    edges.foreach(
      result += _
    )
    result
  }

}