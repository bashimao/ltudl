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

/**
  * As described in:
  * http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
  *
  * Note that the paper uses n_j + n_j+1. However, at module scope we do not
  * know what the actual input fan of the next module will be. So this may
  * be slightly off when applying it to stacked convolutions.
  */
final class XavierGlorotInitializer(override val builder: XavierGlorotInitializerBuilder,
                                    override val seed:    InstanceSeed)
  extends BoostingInitializer[XavierGlorotInitializerBuilder] {
  require(builder != null && seed != null)

  override def computeFanFactor(weights:       ValueTensor,
                                inputFanSize:  Int,
                                outputFanSize: Int)
  : Real = {
    val result = Math.sqrt(2.0 / (inputFanSize + outputFanSize))
    Real(result)
  }

}

final class XavierGlorotInitializerBuilder
  extends BoostingInitializerBuilder[XavierGlorotInitializerBuilder] {

  override def repr
  : XavierGlorotInitializerBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[XavierGlorotInitializerBuilder]

  override def defaultGain()
  : Real = InitializerGain.forLinear

  override protected def doCopy()
  : XavierGlorotInitializerBuilder = XavierGlorotInitializerBuilder()

  override def build(seed: InstanceSeed)
  : XavierGlorotInitializer = new XavierGlorotInitializer(this, seed)

}

object XavierGlorotInitializerBuilder {

  final def apply()
  : XavierGlorotInitializerBuilder = new XavierGlorotInitializerBuilder

  final def apply(source: InitializerBuilder)
  : XavierGlorotInitializerBuilder = apply().setSource(source)

}
