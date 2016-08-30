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
import org.json4s.JsonAST._

object RuntimeStatus {

  final def collect()
  : JObject = {
    logger.trace("Collecting runtime status for edu.latrobe.blaze")

    val fields = List.newBuilder[JField]

    fields += Json.field("plugins", {
      val fields = List.newBuilder[JField]
      Plugin.plugins.foreach(plugin => {
        try {
          fields += Json.field(plugin.name, plugin.collectRuntimeStatus())
        }
        catch {
          case e: Exception =>
            fields += Json.field(plugin.name, s"ERROR: $e")
        }
      })
      JObject(fields.result())
    })

    JObject(fields.result())
  }

}
