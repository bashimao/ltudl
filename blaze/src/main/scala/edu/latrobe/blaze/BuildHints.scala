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

import edu.latrobe._
import edu.latrobe.io.graph.{Edge, LineStyle, Vertex}
import edu.latrobe.sizes._

import scala.util.hashing._

/**
 * An immutable helper structure that provides hints that aid variant selection.
 */
final class BuildHints(val platform:           Platform,
                       val layout:             TensorLayout,
                       val referencePlatform:  Platform,
                       val referenceLayout:    TensorLayout,
                       val preferredPlatform:  Option[Platform])
  extends Equatable
    with Serializable {
  require(
    platform          != null &&
    layout            != null &&
    referencePlatform != null &&
    referenceLayout   != null &&
    preferredPlatform != null
  )

  override def toString: String = {
    s"$layout/$platform/$preferredPlatform, $referenceLayout/$referencePlatform"
  }

  override def hashCode(): Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, platform.hashCode())
    tmp = MurmurHash3.mix(tmp, layout.hashCode())
    tmp = MurmurHash3.mix(tmp, referencePlatform.hashCode())
    tmp = MurmurHash3.mix(tmp, referenceLayout.hashCode())
    tmp = MurmurHash3.mix(tmp, preferredPlatform.hashCode())
    tmp
  }

  override def canEqual(that: Any): Boolean = that.isInstanceOf[BuildHints]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: BuildHints =>
      platform          == other.platform          &&
      layout            == other.layout            &&
      referencePlatform == other.referencePlatform &&
      referenceLayout   == other.referenceLayout   &&
      preferredPlatform == other.preferredPlatform
    case _ =>
      false
  })

  def derive(platform: Platform)
  : BuildHints = derive(
    platform,
    layout
  )

  def derive(layout: TensorLayout)
  : BuildHints = derive(
    platform,
    layout
  )

  def derive(platform: Platform,
             layout:   TensorLayout)
  : BuildHints = derive(
    platform,
    layout,
    referencePlatform,
    referenceLayout
  )

  def derive(platform:          Platform,
             layout:            TensorLayout,
             referencePlatform: Platform,
             referenceLayout:   TensorLayout)
  : BuildHints = BuildHints(
    platform,
    layout,
    referencePlatform,
    referenceLayout,
    preferredPlatform
  )

  def derive(input: Tensor)
  : BuildHints = BuildHints(
    input.platform,
    input.layout,
    referencePlatform,
    referenceLayout,
    preferredPlatform
  )

  def derive(input:     Tensor,
             reference: Tensor)
  : BuildHints = derive(
    input.platform,
    input.layout,
    reference.platform,
    reference.layout
  )

  def withPreferenceFor(preferredPlatform: Platform)
  : BuildHints = withPreferenceFor(Option(preferredPlatform))

  def withPreferenceFor(preferredPlatform: Option[Platform])
  : BuildHints = BuildHints(
    platform,
    layout,
    referencePlatform,
    referenceLayout,
    preferredPlatform
  )


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  def toEdgeLabel
  : String = s"${platform.toEdgeLabel}\n${layout.toEdgeLabel}"

}

object BuildHints {

  final val zero
  : BuildHints = apply(
    JVM,
    IndependentTensorLayout.zero
  )

  final def apply(platform: Platform,
                  layout:   TensorLayout)
  : BuildHints = apply(
    platform,
    layout,
    JVM,
    layout.derive(Size1.zero)
  )

  final def apply(platform:          Platform,
                  layout:            TensorLayout,
                  referencePlatform: Platform,
                  referenceLayout:   TensorLayout)
  : BuildHints = apply(
    platform,
    layout,
    referencePlatform,
    referenceLayout,
    None
  )

  final def apply(platform:          Platform,
                  layout:            TensorLayout,
                  referencePlatform: Platform,
                  referenceLayout:   TensorLayout,
                  preferredPlatform: Option[Platform])
  : BuildHints = new BuildHints(
    platform,
    layout,
    referencePlatform,
    referenceLayout,
    preferredPlatform
  )

  final def derive(input: Tensor)
  : BuildHints = apply(
    input.platform,
    input.layout
  )

  final def derive(input: Tensor, reference: Tensor)
  : BuildHints = apply(
    input.platform,
    input.layout,
    JVM,
    input.layout.derive(Size1.zero)
  )

  final def derive(batch: Batch)
  : BuildHints = derive(batch.input, batch.output)

}
