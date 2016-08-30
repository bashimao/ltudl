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

final class StdErrSink(override val builder: StdErrSinkBuilder,
                       override val seed:    InstanceSeed)
  extends StreamBackedSink[StdErrSinkBuilder] {

  override def write(src0: Any)
  : Unit = src0 match {
    case src0: String =>
      System.err.print(src0)
    case src0: Array[Byte] =>
      System.err.print(Base64.encodeBase64String(src0))
    case src0: JValue =>
      System.err.print(JsonMethods.compact(src0))
    case src0: JsonSerializable =>
      System.err.print(JsonMethods.compact(src0.toJson))
    case _ =>
      System.err.print(src0.toString)
  }

  override def writeRaw(src0: Array[Byte])
  : Unit = System.err.write(src0)

  override def writeRaw(src0: JSerializable)
  : Unit = System.err.write(ArrayEx.serialize(src0))

}

final class StdErrSinkBuilder
  extends StreamBackedSinkBuilder[StdErrSinkBuilder] {

  override def repr
  : StdErrSinkBuilder = this

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[StdErrSinkBuilder]

  override protected def doCopy()
  : StdErrSinkBuilder = StdErrSinkBuilder()

  override def build(seed: InstanceSeed)
  : StdErrSink = new StdErrSink(this, seed)

}

object StdErrSinkBuilder {

  final def apply()
  : StdErrSinkBuilder = new StdErrSinkBuilder

}
