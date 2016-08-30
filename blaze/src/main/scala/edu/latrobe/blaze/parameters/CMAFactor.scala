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
import scala.util.hashing._

/**
  * Factor for computing the cumulative moving average.
  *
  * y = 1 / (1 + n)
  */
final class CMAFactor(override val builder: CMAFactorBuilder,
                      override val name:    String,
                      override val seed:    InstanceSeed)
  extends IndependentParameter[CMAFactorBuilder] {

  override def get(phaseNo: Long)
  : Real = Real.one / (1L + phaseNo)

  override def update(phaseNo: Long, value:  Real)
  : Unit = {}

}

final class CMAFactorBuilder
  extends IndependentParameterBuilder[CMAFactorBuilder] {

  override def repr
  : CMAFactorBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[CMAFactorBuilder]


  override protected def doCopy()
  : CMAFactorBuilder = CMAFactorBuilder()

  override def build(name: String, seed: InstanceSeed)
  : CMAFactor = new CMAFactor(this, name, seed)

}

object CMAFactorBuilder {

  final def apply()
  : CMAFactorBuilder = new CMAFactorBuilder

}
