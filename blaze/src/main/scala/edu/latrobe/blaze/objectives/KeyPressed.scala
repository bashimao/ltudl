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

package edu.latrobe.blaze.objectives

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.io._
import edu.latrobe.time._
import scala.util.hashing._

/**
 * A pseudo target that does nothing except printing the progress figures to
 * the console.
 */
final class KeyPressed(override val builder: KeyPressedBuilder,
                       override val seed:    InstanceSeed)
  extends BinaryTriggerObjective[KeyPressedBuilder](
    ObjectiveEvaluationResult.Neutral
  ) {
  require(builder != null && seed != null)

  val key
  : Int = builder.key

  override protected def doEvaluate(optimizer:           OptimizerLike,
                                    runBeginIterationNo: Long,
                                    runBeginTime:        Timestamp,
                                    runNoSamples:        Long,
                                    model:               Module,
                                    batch:               Batch,
                                    output:              Tensor,
                                    value:               Real)
  : Boolean = LazyKeyboard.keyPressed(key)

}

final class KeyPressedBuilder
  extends BinaryTriggerObjectiveBuilder[KeyPressedBuilder] {

  override def repr
  : KeyPressedBuilder = this

  var key
  : Int = 'q'

  def setKey(value: Int)
  : KeyPressedBuilder = {
    key_=(value)
    this
  }

  override protected def doToString()
  : List[Any] = key :: super.doToString()

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), key.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[KeyPressedBuilder]

  override protected def doEquals(other: Equatable): Boolean = {
    super.doEquals(other)
    other match {
      case other: KeyPressedBuilder =>
        key == other.key
      case _ =>
        false
    }
  }

  override protected def doCopy()
  : KeyPressedBuilder = KeyPressedBuilder()

  override def copyTo(other: InstanceBuilder)
  : Unit = {
    super.copyTo(other)
    other match {
      case other: KeyPressedBuilder =>
        other.key = key
      case _ =>
    }
  }

  override def build(seed: InstanceSeed)
  : KeyPressed = new KeyPressed(this, seed)

}

object KeyPressedBuilder {

  final def apply()
  : KeyPressedBuilder = new KeyPressedBuilder

  final def apply(key: Int)
  : KeyPressedBuilder = apply().setKey(key)

}