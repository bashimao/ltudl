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

import edu.latrobe._
import edu.latrobe.io._
import java.io.BufferedInputStream
import org.apache.http.client.methods._
import org.apache.http.entity._
import org.json4s.JsonAST._
import org.json4s.StreamInput
import org.json4s.jackson.JsonMethods

import scala.collection._

final class Notebook(val id: Long) {

  // Clear the current notebook.
  def clear()
  : Unit = {
    frames.clear()

    val req = new HttpDelete(
      s"http://$LTU_IO_SHOWOFF_HOST_ADDRESS:$LTU_IO_SHOWOFF_HOST_PORT/api/notebook/$id/frames"
    )

    try {
      // HTTP Client >= 4.3 => using(HttpClient.default.execute(req))(rsp => {
      val rsp = HttpClient.default.execute(req)
      if (logger.isInfoEnabled) {
        logger.info(s"Showoff[$id]: Clear => ${rsp.getStatusLine}")
      }
    }
    catch {
      case e: Exception =>
        logger.error(s"Showoff[$id] Clear => $e")
    }
    finally {
      req.abort()
    }
  }

  // TODO: Overload right interfaces and use set.
  private val frames
  : mutable.Map[String, Frame] = mutable.Map.empty

  def getOrCreateFrame(handle: String)
  : Frame = synchronized {
    frames.getOrElseUpdate(handle, {
      // Create json request.
      val json = {
        val tmp = JObject(
          Json.field(
            "frame",
            Json.field("notebookId", id),
            Json.field("type", "text"),
            Json.field("title", s"New Frame - $handle"),
            Json.field(
              "content",
              Json.field("body", "No data present!")
            )
          )
        )
        JsonMethods.compact(tmp)
      }
      logger.trace(json)

      // Post it.
      val req = new HttpPost(
        s"http://$LTU_IO_SHOWOFF_HOST_ADDRESS:$LTU_IO_SHOWOFF_HOST_PORT/api/frame"
      )
      req.setEntity(new StringEntity(json, ContentType.APPLICATION_JSON))

      // Retrieve ID.
      try {
        // HTTP Client >= 4.3 => using(HttpClient.default.execute(req))(rsp => {
        val rsp = HttpClient.default.execute(req)
        if (logger.isInfoEnabled) {
          logger.info(s"Showoff[]: Created frame => ${rsp.getStatusLine}")
        }

        val json = using(
          new BufferedInputStream(rsp.getEntity.getContent)
        )(stream => JsonMethods.parse(StreamInput(stream)))

        // Get frame ID.
        val fields = json.asInstanceOf[JObject].obj.toMap
        val id = Json.toLong(fields("id"))
        if (logger.isInfoEnabled) {
          logger.info(s"Showoff[]: Frame ID = $id")
        }
        Frame(this, id)
      }
      catch {
        case e: Exception =>
          logger.error(s"Showoff | Error = $e")
          Frame(this, -1L)
      }
      finally {
        req.abort()
      }
    })
  }

}

object Notebook {

  final private val notebooks
  : mutable.Map[Int, Notebook] = mutable.Map.empty

  final def get(agentNo: Int)
  : Notebook = notebooks.getOrElseUpdate(agentNo, {
    // Create json request.
    val json = {
      val tmp = JObject(
        Json.field(
          "notebook",
          Json.field("title", s"${Host.name} #$agentNo")
        )
      )
      JsonMethods.compact(tmp)
    }
    logger.trace(json)

    // Post it.
    val req = new HttpPost(
      s"http://$LTU_IO_SHOWOFF_HOST_ADDRESS:$LTU_IO_SHOWOFF_HOST_PORT/api/notebook"
    )
    req.setEntity(new StringEntity(json, ContentType.APPLICATION_JSON))

    // Get notebook ID.
    val id = try {
      // HTTP Client >= 4.3 => using(HttpClient.default.execute(req))(rsp => {
      val rsp = HttpClient.default.execute(req)
      if (logger.isInfoEnabled) {
        logger.info(s"Showoff[] Create Notebook => ${rsp.getStatusLine}")
      }

      val json = using(
        new BufferedInputStream(rsp.getEntity.getContent)
      )(stream => JsonMethods.parse(StreamInput(stream)))

      val fields = json.asInstanceOf[JObject].obj.toMap
      Json.toLong(fields("id"))
    }
    catch {
      case e: Exception =>
        logger.error(s"Showoff[] Create Notebook => $e")
        -1L
    }
    finally {
      req.abort()
    }

    // Create the notebook.
    if (logger.isInfoEnabled) {
      logger.info(s"Showoff[] Notebook ID is $id")
    }
    new Notebook(id)
  })

}