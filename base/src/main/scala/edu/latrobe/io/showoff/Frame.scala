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

package edu.latrobe.io.showoff

import edu.latrobe.io._
import org.apache.http.client.methods._
import org.apache.http.entity._
import org.json4s.jackson.JsonMethods
import edu.latrobe._
import edu.latrobe.io.vega._
import org.json4s.JsonAST._

final class Frame private (val notebook: Notebook, val id: Long) {

  def render(title: String, format: String, content: Any)
  : Unit = {
    try {
      synchronized {
        val actualFormat = {
          if (format == "best") {
            content match {
              case content: Chart =>
                "vega"
              case _ =>
                "text"
            }
          }
          else {
            format
          }
        }

        // Render JSON.
        val json = JObject(
          Json.field(
            "frame",
            Json.field("notebookId", notebook.id),
            Json.field("title", title),
            Json.field("type", actualFormat),
            Json.field(
              "content",
              actualFormat match {
                case "html" =>
                  Json.field("body", content.toString)
                case "text" =>
                  Json.field("body", content.toString)
                case "vega" =>
                  Json.field("body", {
                    content match {
                      case content: JsonSerializable =>
                        content.toJson
                      case content: JValue =>
                        content
                      case _ =>
                        Json(content.toString)
                    }
                  })
                case _ =>
                  throw new MatchError(format)
              }
            )
          )
        )
        val data = JsonMethods.compact(json)
        logger.trace(data)
        //logger.debug(data)

        // Sed request.
        val req = new HttpPut(
          s"http://$LTU_IO_SHOWOFF_HOST_ADDRESS:$LTU_IO_SHOWOFF_HOST_PORT/api/frame/$id"
        )
        req.setEntity(new StringEntity(data, ContentType.APPLICATION_JSON))

        // Begin a HTTP put.
        try {
          // HTTP Client >= 4.3 => using(HttpClient.default.execute(req))(rsp => {
          val rsp = HttpClient.default.execute(req)
          if (logger.isInfoEnabled) {
            logger.info(
              s"Showoff[$id/$title] Update Frame => ${rsp.getStatusLine}"
            )
          }
        }
        finally {
          req.reset()
        }
      }
    }
    catch {
      case e: Exception =>
        logger.error(s"Showoff[$id/$title] Update Frame => $e")
    }
  }

}

private[showoff] object Frame {

  final def apply(notebook: Notebook, id: Long)
  : Frame = new Frame(notebook, id)

}
