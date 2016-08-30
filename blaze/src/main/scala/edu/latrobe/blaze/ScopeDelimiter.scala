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

abstract class ScopeDelimiter
  extends InstanceEx[ScopeDelimiterBuilder]
    with ParameterizedInstance {

  /**
    * @param phaseNo Depends on the context of the environment in which the
    *                scope is used.
    *                Either iterationNo, runNo, combineNo, etc.
    * @return
    */
  def get(phaseNo: Long)
  : NullBuffer

  def update(phaseNo: Long, value: Real)
  : Unit

}

abstract class ScopeDelimiterBuilder
  extends InstanceExBuilder1[ScopeDelimiterBuilder, ScopeDelimiter, NullBuffer] {

  override def build(source: NullBuffer, seed: InstanceSeed)
  : ScopeDelimiter

  final def build(scope: BufferLike, seed: InstanceSeed)
  : ScopeDelimiter = {
    val sourceScope = NullBuffer.derive(scope)
    build(sourceScope, seed)
  }

}

abstract class ScopeDelimiterEx[TBuilder <: ScopeDelimiterExBuilder[_]]
  extends ScopeDelimiter {

  override def builder
  : TBuilder

}

abstract class ScopeDelimiterExBuilder[TThis <: ScopeDelimiterExBuilder[_]]
  extends ScopeDelimiterBuilder {

  override def repr
  : TThis

  override protected def doCopy()
  : TThis

}
