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
import scala.collection._

trait JsonSerializable {

  final def toJson
  : JObject = JObject(doToJson())

  protected def doToJson()
  : List[JField]

}

trait JsonSerializableCompanion {

  def derive(json: JValue)
  : Any

  def derive(json: JObject)
  : Any

}

trait JsonSerializableCompanionEx[T]
  extends JsonSerializableCompanion {

  final override def derive(json: JValue)
  : T = derive(json.asInstanceOf[JObject])

  final override def derive(json: JObject)
  : T = derive(json.obj.toMap)

  def derive(fields: Map[String, JValue])
  : T

}