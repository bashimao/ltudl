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

package edu.latrobe.blaze.modules

import edu.latrobe._
import edu.latrobe.blaze._
import scala.collection._

final class RandomPath(override val builder:             RandomPathBuilder,
                       override val inputHints:          BuildHints,
                       override val seed:                InstanceSeed,
                       override val weightBufferBuilder: ValueTensorBufferBuilder)
  extends PathSwitch[RandomPathBuilder] {

  // ---------------------------------------------------------------------------
  //    Forward propagation related.
  // ---------------------------------------------------------------------------
  override protected def doPredictEx(mode:           Mode,
                                     inPlaceAllowed: Boolean,
                                     input:          Tensor,
                                     reference:      Tensor,
                                     onEnter:        OnEnterPredict,
                                     onLeave:        OnLeavePredict)
  : Int = rng.nextInt(children.length)

}

final class RandomPathBuilder
  extends PathSwitchBuilder[RandomPathBuilder] {

  override def repr
  : RandomPathBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[RandomPathBuilder]

  override protected def doCopy()
  : RandomPathBuilder = RandomPathBuilder()


  // ---------------------------------------------------------------------------
  //   Weights / Building related.
  // ---------------------------------------------------------------------------
  override def build(hints:          BuildHints,
                     seed:           InstanceSeed,
                     weightsBuilder: ValueTensorBufferBuilder)
  : RandomPath = new RandomPath(this, hints, seed, weightsBuilder)

}

object RandomPathBuilder {

  final def apply()
  : RandomPathBuilder = new RandomPathBuilder

  final def apply(module0: ModuleBuilder)
  : RandomPathBuilder = apply() += module0

  final def apply(module0: ModuleBuilder,
                  modules: ModuleBuilder*)
  : RandomPathBuilder = apply(module0)  ++= modules

  final def apply(modules: TraversableOnce[ModuleBuilder])
  : RandomPathBuilder = apply() ++= modules

  final def apply(modules: Array[ModuleBuilder])
  : RandomPathBuilder = apply() ++= modules

}
