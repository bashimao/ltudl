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

package edu.latrobe.io

import edu.latrobe._
import java.io._
import org.apache.commons.compress.archivers._
import scala.collection._

object Archive {

  final private val factory
  : ArchiveStreamFactory = new ArchiveStreamFactory

  @inline
  final def open(stream: InputStream)
  : ArchiveInputStream = factory.createArchiveInputStream(stream)

  final def extract(stream: ArchiveInputStream,
                    filter: LocalFileHandle => Boolean)
  : Map[LocalFileHandle, Array[Byte]] = {
    val builder = Map.newBuilder[LocalFileHandle, Array[Byte]]
    var entry   = stream.getNextEntry
    while (entry != null) {
      // TODO: Does this work with all archives?
      if (!entry.isDirectory) {
        val path = LocalFileHandle(entry.getName)
        if (filter(path)) {
          assume(entry.getSize <= ArrayEx.maxSize)
          val buffer = new Array[Byte](entry.getSize.toInt)
          val n      = stream.read(buffer)
          assume(n == buffer.length)
          builder += Tuple2(path, buffer)
        }
      }
      entry = stream.getNextEntry
    }
    builder.result()
  }

}
