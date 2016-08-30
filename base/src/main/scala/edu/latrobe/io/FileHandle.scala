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
import java.nio._
import java.nio.channels._
import java.nio.charset._
import org.apache.commons.io._
import org.json4s.JsonAST._
import scala.util.matching._

abstract class FileHandle
  extends Serializable
    with JsonSerializable
    with Equatable {

  final def fileName
  : String = FilenameUtils.getName(toString)

  final def fileNameWithoutExtension
  : String = FilenameUtils.getBaseName(toString)

  final def extension
  : String = FilenameUtils.getExtension(toString)

  final def matches(pattern: Regex)
  : Boolean = pattern.findFirstMatchIn(toString).isDefined

  def isDirectory
  : Boolean

  def withoutExtension()
  : FileHandle

  def ++(namePart: String)
  : FileHandle


  // ---------------------------------------------------------------------------
  //    Read/Write
  // ---------------------------------------------------------------------------
  final def createStream()
  : OutputStream = createStream(append = false)

  def createStream(append: Boolean)
  : OutputStream

  final def createChannel()
  : WritableByteChannel = createChannel(append = false)

  def createChannel(append: Boolean)
  : WritableByteChannel

  def openStream()
  : InputStream

  def openChannel()
  : ReadableByteChannel

  def readAsArray()
  : Array[Byte]

  def readAsBuffer()
  : ByteBuffer

  final def readLines()
  : Array[String] = readLines(StandardCharsets.US_ASCII)

  final def readLines(charset: Charset)
  : Array[String] = {
    using(
      openStream()
    )(StreamEx.readLines(_, charset))
  }

  final def readText()
  : String = readText(StandardCharsets.US_ASCII)

  final def readText(charset: Charset)
  : String = {
    using(
      openStream()
    )(StreamEx.readText(_, charset))
  }

  final def write(array: Array[Byte])
  : Unit = write(array, append = false)

  final def write(array: Array[Byte], append: Boolean)
  : Unit = {
    using(
      createStream(append)
    )(_.write(array))
  }

  final def write(array:   Array[Byte],
                  offset:  Int,
                  length:  Int)
  : Unit = write(array, offset, length, append = false)

  final def write(array:   Array[Byte],
                  offset:  Int,
                  length:  Int,
                  append:  Boolean)
  : Unit = {
    using(
      createStream(append)
    )(_.write(array, offset, length))
  }

  final def writeLines(lines:   Array[String])
  : Unit = writeLines(lines, StandardCharsets.US_ASCII)

  final def writeLines(lines:   Array[String],
                       charset: Charset)
  : Unit = writeLines(lines, charset, append = false)

  final def writeLines(lines:   Array[String],
                       charset: Charset,
                       append:  Boolean)
  : Unit = {
    using(
      createStream()
    )(StreamEx.writeLines(_, lines, charset))
  }

  final def writeText(text: String)
  : Unit = writeText(text, StandardCharsets.US_ASCII)

  final def writeText(text: String, charset: Charset)
  : Unit = writeText(text, charset, append = false)

  final def writeText(text:    String,
                      charset: Charset,
                      append:  Boolean)
  : Unit = {
    using(
      createStream()
    )(StreamEx.writeText(_, text, charset))
  }

  final def writeJson(json:    JValue,
                      pretty:  Boolean = true)
  : Unit = {
    using(
      createStream()
    )(StreamEx.writeJson(_, json, pretty))
  }


  // ---------------------------------------------------------------------------
  //    Directory services.
  // ---------------------------------------------------------------------------
  final def listDirectories()
  : Array[FileHandle] = listDirectories((depth, handle) => true)

  final def listDirectories(filter:  (Int, FileHandle) => Boolean)
  : Array[FileHandle] = listDirectories(filter, Int.MaxValue)

  final def listDirectories(filter:   (Int, FileHandle) => Boolean,
                            maxDepth: Int)
  : Array[FileHandle] = {
    val builder = Array.newBuilder[FileHandle]
    traverse(
      (depth, handle) => {
        if (filter(depth, handle)) {
          builder += handle
        }
        depth <= maxDepth
      },
      (depth, handle) => {}
    )
    builder.result
  }

  final def listFiles()
  : Array[FileHandle] = listFiles((depth, handle) => true)

  final def listFiles(filter:  (Int, FileHandle) => Boolean)
  : Array[FileHandle] = listFiles(filter, Int.MaxValue)

  final def listFiles(filter:   (Int, FileHandle) => Boolean,
                      maxDepth: Int)
  : Array[FileHandle] = {
    val builder = Array.newBuilder[FileHandle]
    traverse(
      (depth, handle) => depth <= maxDepth,
      (depth, handle) => {
        if (filter(depth, handle)) {
          builder += handle
        }
      }
    )
    builder.result
  }

  final def traverse(onDirectory: (Int, FileHandle) => Boolean,
                     onFile:      (Int, FileHandle) => Unit)
  : Unit = {
    try {
      if (isDirectory) {
        doTraverse(0, onDirectory, onFile)
      }
      else {
        onFile(0, this)
      }
    }
    catch {
      case e: Exception =>
        logger.error(s"FileHandle.traverse => $e")
        if (LTU_IO_FILE_HANDLE_RETHROW_EXCEPTIONS_DURING_TRAVERSE) {
          throw e
        }
    }
  }

  protected def doTraverse(depth:       Int,
                           onDirectory: (Int, FileHandle) => Boolean,
                           onFile:      (Int, FileHandle) => Unit)
  : Unit

}
