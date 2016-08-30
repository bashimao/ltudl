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

package edu.latrobe.blaze.parameters

import edu.latrobe._
import edu.latrobe.blaze._

/**
  * Use this with constant annealing to create exponential decay.
  * i.e. alpha = phi * exp(-k t)
  */
final class Exp(override val builder: ExpBuilder,
                override val name:    String,
                override val seed:    InstanceSeed)
  extends DependentParameter[ExpBuilder] {

  override def get(phaseNo: Long)
  : Real = Real(Math.exp(DoubleEx(super.get(phaseNo))))

}

final class ExpBuilder
  extends DependentParameterBuilder[ExpBuilder] {

  override def repr
  : ExpBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[ExpBuilder]

  override protected def doCopy()
  : ExpBuilder = ExpBuilder()

  override def build(name: String, seed: InstanceSeed)
  : Exp = new Exp(this, name, seed)

}

object ExpBuilder {

  final def apply()
  : ExpBuilder = new ExpBuilder

}
