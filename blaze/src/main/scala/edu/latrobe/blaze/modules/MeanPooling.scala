/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2014 Matthias Langer (t3l@threelights.de)
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
 */

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules.jvm._
import scala.util.hashing._

/**
  * Mean pooling layer.
  *
  * p = position in output
  * m = size covered by kernel at position p
  *
  *               m
  *              ---
  *           1  \
  * f(p_a) = --- /   x_pi
  *           m  ---
  *               i
  *
  * d f(p_a)   1
  * -------- = -
  *  d x_pa    m
  *
  *   d f(p_a)     1
  * ------------ = -
  * d x_pb, a!=b   m
  *
  *            ---
  * D f(p_a)   \   d f(p_a)
  * -------- = /   -------- di
  *  D x_pa    ---  d x_pi
  *             i
  *
  *              ---
  *            1 \
  *          = - /   di
  *            m ---
  *               i
  *
  */
abstract class MeanPooling
  extends PoolingLayerEx[MeanPoolingBuilder] {

  final val includePadding
  : Boolean = builder.includePadding

}

final class MeanPoolingBuilder
  extends PoolingLayerExBuilder[MeanPoolingBuilder] {

  override def repr
  : MeanPoolingBuilder = this

  var includePadding
  : Boolean = true

  def setIncludePadding(value: Boolean)
  : MeanPoolingBuilder = {
    includePadding_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = includePadding :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), includePadding.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[MeanPoolingBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: MeanPoolingBuilder =>
      includePadding == other.includePadding
    case _ =>
      false
  })

  override protected def doCopy()
  : MeanPoolingBuilder = MeanPoolingBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: MeanPoolingBuilder =>
        other.includePadding = includePadding
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(hints: BuildHints)
  : Platform = MeanPoolingBuilder.outputPlatformFor(this, hints)

  // Lookup variant and create object.
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = MeanPoolingBuilder.lookupAndBuild(
    this, hints, seed, weightsBuilder
  )

}

object MeanPoolingBuilder
  extends ModuleVariantTable[MeanPoolingBuilder] {

  register(2, MeanPooling_JVM_Baseline_Description)

  final def apply()
  : MeanPoolingBuilder = new MeanPoolingBuilder

  final def apply(kernel: Kernel)
  : MeanPoolingBuilder = apply().setKernel(kernel)

  final def apply(kernel: Kernel, includePadding: Boolean)
  : MeanPoolingBuilder = apply(kernel).setIncludePadding(includePadding)

}
