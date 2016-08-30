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
import java.io.{File, InputStream, OutputStream}
import java.nio.channels._
import java.nio.charset._
import java.nio.file._
import java.nio.file.attribute._

object LocalFileSystem {

  /*
  final private val fs
  : FileSystem = FileSystems.getDefault

  @inline
  final def open(path: LocalPath)
  : FileChannel = {
    val options = new java.util.HashSet[OpenOption](1)
    options.add(StandardOpenOption.READ)

    fs.provider().newByteChannel(path, options)
  }

  @inline
  final def openStream(file: File)
  : InputStream = openStream(file.toPath)

  @inline
  final def openStream(path: LocalPath)
  : InputStream = Channels.newInputStream(open(path))

  @inline
  final def create(path: LocalPath)
  : SeekableByteChannel = {
    val options = new java.util.HashSet[OpenOption](3)
    options.add(StandardOpenOption.WRITE)
    options.add(StandardOpenOption.CREATE)
    options.add(StandardOpenOption.TRUNCATE_EXISTING)
    fs.provider().newByteChannel(path, options)
  }

  @inline
  final def createStream(file: File)
  : OutputStream = createStream(file.toPath)

  @inline
  final def createStream(path: LocalPath)
  : OutputStream = Channels.newOutputStream(create(path))
  */


  /*
  @inline
  final def read(path: LocalPath): Array[Byte] = {
    using(open(path))(channel => {
      val sizeHint = channel.size()
      if (sizeHint > ArrayEx.maxSize) {
        throw new OutOfMemoryError("File larger than maximum array size.")
      }
      using(Channels.newInputStream(channel))(
        StreamEx.read(_, sizeHint.toInt)
      )
    })
  }
  */

  /*
  @inline
  final def readLines(path:    LocalPath,
                      charset: Charset = StandardCharsets.US_ASCII)
  : Array[String] = {
    using(openStream(path))(
      StreamEx.readLines(_, charset)
    )
  }

  @inline
  final def readText(path:    LocalPath,
                     charset: Charset = StandardCharsets.US_ASCII)
  : String = {
    using(openStream(path))(
      StreamEx.readText(_, charset)
    )
  }
  */

  /*
  @inline
  final def write(file:  File,
                  bytes: Array[Byte])
  : Unit = write(file, bytes, 0, bytes.length)

  @inline
  final def write(file:  File,
                  bytes: Array[Byte], offset: Int, length: Int)
  : Unit = {
    using(
      createStream(file)
    )(_.write(bytes, offset, length))
  }
  */

  /*
  @inline
  final def write(path:  LocalPath,
                  bytes: Array[Byte])
  : Unit = write(path, bytes, 0, bytes.length)

  @inline
  final def write(path:  LocalPath,
                  bytes: Array[Byte], offset: Int, length: Int)
  : Unit = {
    using(
      createStream(path)
    )(_.write(bytes, offset, length))
  }

  @inline
  final def writeLines(path:    LocalPath,
                       lines:   Array[String],
                       charset: Charset = StandardCharsets.US_ASCII)
  : Unit = {
    using(createStream(path))(
      StreamEx.writeLines(_, lines, charset)
    )
  }

  @inline
  final def writeText(path:    LocalPath,
                      text:    String,
                      charset: Charset = StandardCharsets.US_ASCII)
  : Unit = {
    using(createStream(path))(
      StreamEx.writeText(_, text, charset)
    )
  }
  */

}
