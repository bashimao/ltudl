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
import edu.latrobe.blaze.modules.generic._
import edu.latrobe.blaze.modules.jvm._
import edu.latrobe.blaze.TensorDependency._
import scala.util.hashing._

/**
  * L2 pooling layer
  *
  * This is a slight variation of what I have found on the internet. Used divide
  * by size(N) to avoid overflows of y as N grows and keep downstream values scaled
  * nicely. It also makes the values easier comparable with other pooling layers.
  *
  * For the a'th kernel are coordinates in a i-th finite m-shaped subspace.
  *
  *                 /----------------
  *                /        m
  *               /        ---
  * f(p_a) =     /       1 \       2
  *             /    e + - /   x_pi
  *            /         m ---
  *          \/             i
  *
  * The epsilon parameter can be used to make the L2 pooling smooth. Otherwise,
  * L2 pooling may not pass gradient checks.
  *
  *                          (       ---       )
  *                          (     1 \       2 )
  *                        d ( e + - /   x_pi  )
  *                          (     m ---       )
  * d f(p_a)       1         (        i        )
  * -------- = --------- * ---------------------
  *  d x_pa    2 f(x_pa)           d x_pa
  *
  *                1       2 x_pa
  *          = --------- * ------
  *            2 f(x_pa)     m
  *
  *               x_pa
  *          = ---------
  *            f(x_pa) m
  *
  *   d f(p_a)       x_pb
  * ------------ = ---------
  * d x_pb, a!=b   f(x_pa) m
  *
  *
  *            ---
  * D f(p_a)   \   d f(p_a)
  * -------- = /   -------- di
  *  D x_pa    ---  d x_pi
  *             i
  *
  *              ---
  *            1 \     x_pi
  *          = - /   ------- di
  *            m --- f(x_pa)
  *               i
  *
  */
abstract class L2Pooling
  extends PoolingLayerEx[L2PoolingBuilder] {

  final val includePadding
  : Boolean = builder.includePadding

  final val epsilon
  : Real = builder.epsilon


  // ---------------------------------------------------------------------------
  //    Back propagation related.
  // ---------------------------------------------------------------------------
  final override val backpropagationRequirementsForInput
  : TensorDependency = TensorDependency.Required

  final override val backpropagationRequirementsForOutput
  : TensorDependency = TensorDependency.Required

}

final class L2PoolingBuilder
  extends PoolingLayerExBuilder[L2PoolingBuilder] {

  override def repr
  : L2PoolingBuilder = this

  var includePadding
  : Boolean = false

  def setIncludePadding(value: Boolean)
  : L2PoolingBuilder = {
    includePadding_=(value)
    this
  }

  private var _epsilon
  : Real = 1.00000003e-5f

  def epsilon
  : Real = _epsilon

  def epsilon_=(value: Real)
  : Unit = {
    require(value >= Real.zero)
    _epsilon = value
  }

  def setEpsilon(value: Real)
  : L2PoolingBuilder = {
    epsilon_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = includePadding :: f"${_epsilon}%.4g" :: super.doToString()

  override def hashCode()
  : Int = {
    var tmp = super.hashCode()
    tmp = MurmurHash3.mix(tmp, includePadding.hashCode())
    tmp = MurmurHash3.mix(tmp, _epsilon.hashCode())
    tmp
  }

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[L2PoolingBuilder]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: L2PoolingBuilder =>
      includePadding == other.includePadding &&
      _epsilon       == other._epsilon
    case _ =>
      false
  })

  override protected def doCopy()
  : L2PoolingBuilder = L2PoolingBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: L2PoolingBuilder =>
        other.includePadding = includePadding
        other._epsilon       = _epsilon
      case _ =>
    }
  }


  // ---------------------------------------------------------------------------
  //     Weights / binding related
  // ---------------------------------------------------------------------------
  override def outputPlatformFor(hints: BuildHints)
  : Platform = L2PoolingBuilder.outputPlatformFor(this, hints)

  // Lookup variant and create object.
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : Module = L2PoolingBuilder.lookupAndBuild(this, hints, seed, weightsBuilder)

}

object L2PoolingBuilder
  extends ModuleVariantTable[L2PoolingBuilder] {

  register( 2, L2Pooling_JVM_Baseline_Description)
  register(64, L2Pooling_Generic_Baseline_Description)

  final def apply(): L2PoolingBuilder = new L2PoolingBuilder

  final def apply(kernel: Kernel)
  : L2PoolingBuilder = apply().setKernel(kernel)

  final def apply(kernel: Kernel, includePadding: Boolean)
  : L2PoolingBuilder = apply(kernel).setIncludePadding(includePadding)

  final def apply(kernel: Kernel, includePadding: Boolean, epsilon: Real)
  : L2PoolingBuilder = apply(kernel, includePadding).setEpsilon(epsilon)

}
