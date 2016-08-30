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

package edu.latrobe.blaze.scopedelimiters

import edu.latrobe._
import edu.latrobe.blaze._

import scala.collection._
import scala.util.hashing._

final class AlternateSegments(override val builder: AlternateSegmentsBuilder,
                              override val scope:   NullBuffer,
                              override val seed:    InstanceSeed)
  extends IndependentScopeEx[AlternateSegmentsBuilder] {

  private val sequence
  : Array[SimpleBufferReference] = builder.segmentSequence.toArray

  override def get(phaseNo: Long)
  : NullBuffer = {
    val reference = sequence((phaseNo % sequence.length).toInt)
    NullBuffer.derive(reference)
  }

}

final class AlternateSegmentsBuilder
  extends IndependentScopeExBuilder[AlternateSegmentsBuilder] {

  override def repr
  : AlternateSegmentsBuilder = this

  val segmentSequence
  : mutable.Buffer[SimpleBufferReference] = mutable.Buffer.empty

  def +=(reference: SimpleBufferReference)
  : AlternateSegmentsBuilder = {
    segmentSequence += reference
    this
  }

  def ++=(references: TraversableOnce[SimpleBufferReference])
  : AlternateSegmentsBuilder = {
    segmentSequence ++= references
    this
  }

  override protected def doToString()
  : List[Any] = segmentSequence.length :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), segmentSequence.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[AlternateSegmentsBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: AlternateSegmentsBuilder =>
      segmentSequence == other.segmentSequence
    case _ =>
      false
  })

  override protected def doCopy()
  : AlternateSegmentsBuilder = AlternateSegmentsBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: AlternateSegmentsBuilder =>
        other.segmentSequence.clear()
        other.segmentSequence ++= segmentSequence
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Instance building related.
  // ---------------------------------------------------------------------------
  override def build(source: NullBuffer,
                     seed:   InstanceSeed)
  : AlternateSegments = new AlternateSegments(this, source, seed)

}

object AlternateSegmentsBuilder {

  final def apply()
  : AlternateSegmentsBuilder = new AlternateSegmentsBuilder

  final def apply(reference0: SimpleBufferReference)
  : AlternateSegmentsBuilder = apply() += reference0

  final def apply(reference0: SimpleBufferReference,
                  referenceN: SimpleBufferReference*)
  : AlternateSegmentsBuilder = apply(reference0) ++= referenceN

  final def apply(references: TraversableOnce[SimpleBufferReference])
  : AlternateSegmentsBuilder = apply() ++= references

}
