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

package edu.latrobe

import org.json4s.JsonAST._
import edu.latrobe.sizes._

import scala.collection.Map

/**
  * This is the abstract base for everything that denotes a "size" of a sample,
 * layer, a kernel, etc. (I know this would not have been necessary, but I
 * wanted to avoid having.
 *
 * Sizes must be immutable!
 */
abstract class Size
  extends Equatable
    with Serializable
    with JsonSerializable {

  def noChannels
  : Int

  /**
   * Hint: This field is used very frequently!
 *
   * @return Number of data tuples of the object.
   */
  def noTuples
  : Int

  /**
   * Hint: This field is used very frequently!
 *
   * @return Number of actual values present (noTuples * noChannels)!
   */
  def noValues
  : Int

  //final def isEmpty: Boolean = noValues == 0

  //final def flatten: Size1 = Size1(noTuples, noChannels)

  def multiplex
  : Size

  def demultiplex
  : Size

  /**
   * Concatenates sizes. (Number of channels must be the same! Shape may be relevant!)
   */
  def ++(other: Size)
  : Size

  /**
   * Concatenates tuples. (Shape of both sizes must be equivalent!)
   */
  def :++(other: Size)
  : Size

  /**
    * Kind of an inverse operation of :++.
    */
  def withNoTuples(noTuples: Int)
  : Size

  /**
    * Kind of an inverse operation of :++.
    */
  def withNoChannels(noChannels: Int)
  : Size

}

abstract class SizeCompanion
  extends JsonSerializableCompanion

object Size
  extends SizeCompanion
    with JsonSerializableCompanionEx[Size] {

  // TODO: Do this with reflection!
  override def derive(fields: Map[String, JValue])
  : Size = {
    val className = Json.toString(fields("className"))
    className match {
      case "Size1" =>
        Size1.derive(fields)
      case "Size2" =>
        Size2.derive(fields)
      case "Size3" =>
        Size3.derive(fields)
      case "Size4" =>
        Size4.derive(fields)
      case _ =>
        throw new MatchError(className)
    }
  }

}

abstract class SizeEx[TThis <: SizeEx[_]]
  extends Size {

  override def :++(other: Size)
  : TThis

  override def withNoTuples(noTuples: Int)
  : TThis

  override def withNoChannels(noChannels: Int)
  : TThis

}

abstract class SizeExCompanion[T <: SizeEx[_]]
  extends SizeCompanion
    with JsonSerializableCompanionEx[T] {
}
