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

abstract class IndependentScope[TBuilder <: IndependentScopeBuilder[_]]
  extends ScopeDelimiterEx[TBuilder] {

  /**
    * Must be implemented as constructor argument!
    */
  def scope
  : NullBuffer

}

abstract class IndependentScopeBuilder[TThis <: IndependentScopeBuilder[_]]
  extends ScopeDelimiterExBuilder[TThis] {
}

abstract class IndependentScopeEx[TBuilder <: IndependentScopeExBuilder[_]]
  extends IndependentScope[TBuilder] {

  final override def update(phaseNo: Long, value: Real)
  : Unit = {}

}

abstract class IndependentScopeExBuilder[TThis <: IndependentScopeExBuilder[_]]
  extends IndependentScopeBuilder[TThis] {
}
