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

package edu.latrobe.blaze.batchpools

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._
import edu.latrobe.io.graph._
import scala.collection._
import scala.util.hashing._

/**
  * Pools that inherit from this can be equipped with modules that work
  * on the input.
  *
  * This class can be used to feed the data directly though a module via
  * "projection" at load time. If the module has weights you will have to
  * provide them to the builder.
  */
abstract class Augmenter[TBuilder <: AugmenterBuilder[_]]
  extends DependentBatchPool[TBuilder] {

  final val mode
  : Mode = builder.mode
}

abstract class AugmenterBuilder[TThis <: AugmenterBuilder[_]]
  extends DependentBatchPoolBuilder[TThis] {

  final private var _mode
  : Mode = Inference()

  final def mode
  : Mode = _mode

  final def mode_=(value: Mode)
  : Unit = {
    require(value != null)
    _mode = value
  }

  final def setMode(value: Mode)
  : TThis = {
    mode_=(value)
    repr
  }

  override protected def doToString()
  : List[Any] = _mode :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), _mode.hashCode())

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: AugmenterBuilder[TThis] =>
      _mode == other._mode
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder): Unit = {
    super.copyTo(other)
    other match {
      case other: AugmenterBuilder[TThis] =>
        other._mode = _mode
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  final override protected def doToGraphEx(hints:    Option[BuildHints],
                                           inputs:   Seq[Vertex],
                                           nodeSink: mutable.Buffer[Node],
                                           edgeSink: mutable.Buffer[Edge])
  : (Option[BuildHints], Seq[Vertex]) = {
    // Create a vertex-group for the augmenter.
    val node = VertexGroup.derive(toString("\n", ""))
    nodeSink += node

    // Connect incoming edges to model.
    doToGraphExEx(hints, inputs, node.children, edgeSink)
  }

  protected def doToGraphExEx(hints:    Option[BuildHints],
                              inputs:   Seq[Vertex],
                              nodeSink: mutable.Buffer[Node],
                              edgeSink: mutable.Buffer[Edge])
  : (Option[BuildHints], Seq[Vertex])

}

final case class AugmenterState(override val parent: InstanceState,
                                module:              InstanceState)
  extends BatchPoolState
