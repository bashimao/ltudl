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

abstract class IndependentPlatform
  extends Platform {

  def handle
  : String

  override protected def doToJson()
  : List[JField] = List(
    Json.field("handle", handle)
  )


  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  final override def toEdgeLabel
  : String = handle

}


object IndependentPlatform
  extends PlatformCompanion
    with JsonSerializableCompanionEx[IndependentPlatform] {

  final private val registered
  : mutable.Map[String, IndependentPlatform] = mutable.Map.empty

  @inline
  final def derive(name: String)
  : Option[IndependentPlatform] = registered.get(name)

  override def derive(fields: Map[String, JValue])
  : IndependentPlatform = registered(Json.toString(fields("handle")))

  def register(platform: IndependentPlatform)
  : Unit = registered += Tuple2(platform.handle, platform)

  /**
    * Use with care!
    */
  def unregister(platform: IndependentPlatform)
  : Unit = registered.remove(platform.handle)

  register(JVM)

  // TODO: Help wanted!
  /*
  case object DirectCompute
    extends PlatformID
      with GPUPlatformID
  */

  // TODO: Help wanted!
  /*
  case object OpenCL
    extends PlatformID
      with GPUPlatformID
  */

}