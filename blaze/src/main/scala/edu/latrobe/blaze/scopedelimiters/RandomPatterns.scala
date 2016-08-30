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

final class RandomPatterns(override val builder: RandomPatternsBuilder,
                           override val scope:   NullBuffer,
                           override val seed:    InstanceSeed)
  extends IndependentScope[RandomPatternsBuilder] {

  private val patterns
  : Array[NullBuffer] = {
    ArrayEx.map(
      scope.references
    )(NullBuffer.derive(_))
  }

  private var patternNo
  : Int = -1

  override def get(phaseNo: Long)
  : NullBuffer = {
    if (patternNo < 0) {
      update(phaseNo, Real.nan)
    }
    patterns(patternNo)
  }

  override def update(phaseNo: Long, value: Real)
  : Unit = patternNo = rng.nextInt(patterns.length)


  // ---------------------------------------------------------------------------
  //    State management.
  // ---------------------------------------------------------------------------
  override def state
  : InstanceState = RandomPatternsState(super.state, patternNo)

  override def restoreState(state: InstanceState)
  : Unit = {
    super.restoreState(state.parent)
    state match {
      case state: RandomPatternsState =>
        patternNo = state.patternNo
      case _ =>
        throw new MatchError(state)
    }
  }

}

final class RandomPatternsBuilder
  extends IndependentScopeBuilder[RandomPatternsBuilder] {

  override def repr
  : RandomPatternsBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[RandomPatternsBuilder]

  override protected def doCopy()
  : RandomPatternsBuilder = RandomPatternsBuilder()


  // ---------------------------------------------------------------------------
  //    Instance building related.
  // ---------------------------------------------------------------------------
  override def build(source: NullBuffer,
                     seed:   InstanceSeed)
  : RandomPatterns = new RandomPatterns(this, source, seed)

}

object RandomPatternsBuilder {

  final def apply()
  : RandomPatternsBuilder = new RandomPatternsBuilder

}

final case class RandomPatternsState(override val parent: InstanceState,
                                     patternNo:           Int)
  extends InstanceState {
}
