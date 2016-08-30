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

import scala.collection._

abstract class Platform
  extends Serializable
    with JsonSerializable {

  // ---------------------------------------------------------------------------
  //    Conversion related
  // ---------------------------------------------------------------------------
  def toEdgeLabel
  : String

}

object Platform {

  final private var toFunctions
  : mutable.Map[Platform, (Tensor, Any) => Tensor] = mutable.Map.empty

  final private var asOrToFunctions
  : mutable.Map[Platform, (Tensor, Any) => Tensor] = mutable.Map.empty

  final private[latrobe] def register(platform:  Platform,
                                      toFn:      (Tensor, Any) => Tensor,
                                      asOrToFn:  (Tensor, Any) => Tensor)
  : Unit = {
    toFunctions     += Tuple2(platform, toFn)
    asOrToFunctions += Tuple2(platform, asOrToFn)
  }

  final private[latrobe] def unregister(platform: Platform)
  : Unit = {
    toFunctions.remove(platform)
    asOrToFunctions.remove(platform)
  }

  final def to(platform: Platform, tensor: Tensor, context: Any)
  : Tensor = {
    val fn = toFunctions(platform)
    fn(tensor, context)
  }

  final def asOrTo(platform: Platform, tensor: Tensor, context: Any)
  : Tensor = {
    val fn = toFunctions(platform)
    fn(tensor, context)
  }

  register(
    JVM,
    (tensor, context) => tensor.toRealArrayTensor,
    (tensor, context) => tensor.asOrToRealArrayTensor
  )

}

abstract class PlatformCompanion
  extends JsonSerializableCompanion {
}
