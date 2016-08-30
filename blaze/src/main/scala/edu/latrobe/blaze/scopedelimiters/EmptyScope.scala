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

final class EmptyScope(override val builder: EmptyScopeBuilder,
                       override val scope:   NullBuffer,
                       override val seed:    InstanceSeed)
  extends IndependentScopeEx[EmptyScopeBuilder] {

  override def get(phaseNo: Long)
  : NullBuffer = NullBuffer.empty

}

final class EmptyScopeBuilder
  extends IndependentScopeExBuilder[EmptyScopeBuilder] {

  override def repr
  : EmptyScopeBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[EmptyScopeBuilder]

  override protected def doCopy()
  : EmptyScopeBuilder = EmptyScopeBuilder()

  override def build(source: NullBuffer,
                     seed:   InstanceSeed)
  : EmptyScope = new EmptyScope(this, source, seed)

}

object EmptyScopeBuilder {

  final def apply()
  : EmptyScopeBuilder = new EmptyScopeBuilder

}
