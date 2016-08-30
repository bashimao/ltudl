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
import scala.util.hashing._

/**
  * Inference mode is typically used for prediction once a model has been
  * trained, or for cross validation during training.
  */
final class Inference(override val reproducible: Boolean)
  extends Mode {

  override def toString
  : String = s"Inference[$reproducible]"

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), reproducible.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[Inference]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: Inference =>
      reproducible == other.reproducible
    case _ =>
      false
  })

  override def supportsBackpropagation
  : Boolean = false

}

object Inference {

  final def apply()
  : Inference = apply(reproducible = false)

  final def apply(reproducible: Boolean)
  : Inference = new Inference(reproducible)

  final def reproducible()
  : Inference = apply(reproducible = true)

}
