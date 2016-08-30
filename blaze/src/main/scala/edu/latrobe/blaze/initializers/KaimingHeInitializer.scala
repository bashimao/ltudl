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

package edu.latrobe.blaze.initializers

import edu.latrobe._
import edu.latrobe.blaze._
import scala.util.hashing._

/**
  * Recommended function for initializing filters preceeding ReLUs.
  *
  * As described in:
  * http://arxiv-web3.library.cornell.edu/abs/1502.01852
  *
  * The torch default initializer always keeps gain at 1.
  */
final class KaimingHeInitializer(override val builder: KaimingHeInitializerBuilder,
                                 override val seed:    InstanceSeed)
  extends BoostingInitializer[KaimingHeInitializerBuilder] {
  require(builder != null && seed != null)

  val useOutputFanSize
  : Boolean = builder.useOutputFanSize

  override def computeFanFactor(weights:       ValueTensor,
                                inputFanSize:  Int,
                                outputFanSize: Int)
  : Real = {
    val fanSize = if (useOutputFanSize) outputFanSize else inputFanSize
    val result = Math.sqrt(1.0 / fanSize)
    Real(result)
  }

}

final class KaimingHeInitializerBuilder
  extends BoostingInitializerBuilder[KaimingHeInitializerBuilder] {

  override def repr
  : KaimingHeInitializerBuilder = this

  override def defaultGain()
  : Real = InitializerGain.forLinear

  var useOutputFanSize
  : Boolean = false

  def setUseOutputFanSize(value: Boolean)
  : KaimingHeInitializerBuilder = {
    useOutputFanSize_=(value)
    this
  }

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), useOutputFanSize.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[KaimingHeInitializerBuilder]

  override protected def doCopy()
  : KaimingHeInitializerBuilder = KaimingHeInitializerBuilder()

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: KaimingHeInitializerBuilder =>
      useOutputFanSize == other.useOutputFanSize
    case _ =>
      false
  })

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: KaimingHeInitializerBuilder =>
        other.useOutputFanSize = useOutputFanSize
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : KaimingHeInitializer = new KaimingHeInitializer(this, seed)

}

object KaimingHeInitializerBuilder {

  final def apply(): KaimingHeInitializerBuilder = new KaimingHeInitializerBuilder

  final def apply(gain: Real)
  : KaimingHeInitializerBuilder = apply().setGain(gain)

  final def apply(gain: Real, source: InitializerBuilder)
  : KaimingHeInitializerBuilder = apply(gain).setSource(source)

}