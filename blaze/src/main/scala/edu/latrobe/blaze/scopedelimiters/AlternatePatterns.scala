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

final class AlternatePatterns(override val builder: AlternatePatternsBuilder,
                              override val scope:   NullBuffer,
                              override val seed:    InstanceSeed)
  extends IndependentScopeEx[AlternatePatternsBuilder] {

  private val patternSequence
  : Array[NullBuffer] = builder.patternSequence.toArray

  override def get(phaseNo: Long)
  : NullBuffer = {
    val index = phaseNo % patternSequence.length
    patternSequence(index.toInt)
  }

}

final class AlternatePatternsBuilder
  extends IndependentScopeExBuilder[AlternatePatternsBuilder] {

  override def repr
  : AlternatePatternsBuilder = this

  val patternSequence
  : mutable.Buffer[NullBuffer] = mutable.Buffer.empty

  def +=(pattern: NullBuffer)
  : AlternatePatternsBuilder = {
    patternSequence += pattern
    this
  }

  def ++=(patterns: TraversableOnce[NullBuffer])
  : AlternatePatternsBuilder = {
    patternSequence ++= patterns
    this
  }

  def addPatterns(patterns: Seq[NullBuffer], indices: Seq[Int])
  : AlternatePatternsBuilder = {
    indices.foreach(
      index => patternSequence += patterns(index)
    )
    this
  }

  override protected def doToString()
  : List[Any] = patternSequence.length :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), patternSequence.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[AlternatePatternsBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: AlternatePatternsBuilder =>
      patternSequence == other.patternSequence
    case _ =>
      false
  })

  override protected def doCopy()
  : AlternatePatternsBuilder = AlternatePatternsBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: AlternatePatternsBuilder =>
        other.patternSequence.clear()
        other.patternSequence ++= patternSequence
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //    Instance building related.
  // ---------------------------------------------------------------------------
  override def build(source: NullBuffer,
                     seed:   InstanceSeed)
  : AlternatePatterns = new AlternatePatterns(this, source, seed)

}

object AlternatePatternsBuilder {

  final def apply()
  : AlternatePatternsBuilder = new AlternatePatternsBuilder

  final def apply(pattern0: NullBuffer)
  : AlternatePatternsBuilder = apply() += pattern0

  final def apply(pattern0: NullBuffer,
                  patternN: NullBuffer*)
  : AlternatePatternsBuilder = apply(pattern0) ++= patternN

  final def apply(patterns: TraversableOnce[NullBuffer])
  : AlternatePatternsBuilder = apply() ++= patterns

  final def apply(patterns: Seq[NullBuffer], indices: Seq[Int])
  : AlternatePatternsBuilder = apply().addPatterns(patterns, indices)

}
