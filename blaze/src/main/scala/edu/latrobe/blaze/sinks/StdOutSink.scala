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

package edu.latrobe.blaze.sinks

import edu.latrobe._
import edu.latrobe.blaze._
import org.apache.commons.codec.binary._
import org.json4s.JsonAST._
import org.json4s.jackson._

final class StdOutSink(override val builder: StdOutSinkBuilder,
                       override val seed:    InstanceSeed)
  extends StreamBackedSink[StdOutSinkBuilder] {

  override def write(src0: Any)
  : Unit = src0 match {
    case src0: String =>
      System.out.print(src0)
    case src0: Array[Byte] =>
      System.out.print(Base64.encodeBase64String(src0))
    case src0: JValue =>
      System.out.print(JsonMethods.compact(src0))
    case src0: JsonSerializable =>
      System.out.print(JsonMethods.compact(src0.toJson))
    case _ =>
      System.out.print(src0.toString)
  }

  override def writeRaw(src0: Array[Byte])
  : Unit = System.out.write(src0)

  override def writeRaw(src0: JSerializable)
  : Unit = System.err.write(ArrayEx.serialize(src0))

}

final class StdOutSinkBuilder
  extends StreamBackedSinkBuilder[StdOutSinkBuilder] {

  override def repr
  : StdOutSinkBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[StdOutSinkBuilder]

  override protected def doCopy()
  : StdOutSinkBuilder = StdOutSinkBuilder()

  override def build(seed: InstanceSeed)
  : StdOutSink = new StdOutSink(this, seed)

}

object StdOutSinkBuilder {

  final def apply()
  : StdOutSinkBuilder = new StdOutSinkBuilder

}
