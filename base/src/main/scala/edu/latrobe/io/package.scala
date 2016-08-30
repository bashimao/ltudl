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

import java.nio._
import java.nio.file._
import org.json4s.JsonAST._
import org.json4s.jackson._
import org.bytedeco.javacpp._

/**
  * Core stuff of blaze & inferno that I do not want to clutter the namespace.
  */
package object io {

  /*
  final implicit class IOLocalPathFunctions(lp: LocalPath) {

    /**
      * Appends a directory.
      */
    @inline
    def ++(other: String): LocalPath = LocalPath.concat(lp, other)

    /**
      * Concatenates two path fractions.
      */
    @inline
    def ++(other: LocalPath): LocalPath = LocalPath.concat(lp, other)

  }

  type URI = java.net.URI

  final implicit class IOURIFunctions(uri: URI) {

    /**
      * Appends a directory.
      */
    @inline
    def ++(other: String): URI = new URI(
      uri.getScheme,
      uri.getUserInfo,
      uri.getHost,
      uri.getPort,
      if (uri.getPath.isEmpty && other != "/") {
        Paths.get("/", uri.getPath, other).normalize().toString
      }
      else {
        Paths.get(uri.getPath, other).normalize().toString
      },
      uri.getQuery,
      uri.getFragment
    )

  }
  */

  /*
  final implicit class IOInputStreamFunctions(str: InputStream) {

    @inline
    def buffer(bufferSize: Int = -1): BufferedInputStream = {
      if (bufferSize == -1) {
        new BufferedInputStream(str)
      }
      else {
        new BufferedInputStream(str, bufferSize)
      }
    }

    @inline
    def toDataInputStream: DataInputStream = new DataInputStream(str)

    @inline
    def toObjectInputStream: ObjectInputStream = new ObjectInputStream(str)

  }
  */

  /*
  final implicit class IOJObjectFunctions(json: JObject) {

    def render(): String = JsonMethods.compact(json)

    def saveAs(path: LocalPath): Unit = render().saveAs(path)

  }
  */

  final val LTU_IO_FILE_HANDLE_RETHROW_EXCEPTIONS_DURING_TRAVERSE
  : Boolean = Environment.parseBoolean(
    "LTU_IO_FILE_HANDLE_RETHROW_EXCEPTIONS_DURING_TRAVERSE",
    default = true
  )

}
