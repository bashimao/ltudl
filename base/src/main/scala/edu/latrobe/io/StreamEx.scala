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

import com.fasterxml.jackson.databind._
import edu.latrobe._
import java.io._
import java.nio.charset._
import org.json4s.JsonAST._
import org.json4s.jackson._

object StreamEx {

  // Works even if the sizeHint is wrong.
  @inline
  final def read(stream: InputStream, sizeHint: Int = 8192)
  : Array[Byte] = {
    var buffer = new Array[Byte](sizeHint)
    var used   = 0
    var n      = -1

    while (true) {
      // Try to read all bytes.
      n = stream.read(buffer, used, buffer.length - used)
      while (n > 0) {
        used += n
        n = stream.read(buffer, used, buffer.length - used)
      }

      // If EOF reached.
      if (n < 0) {
        if (used == buffer.length) {
          return buffer
        }
        else {
          return java.util.Arrays.copyOf(buffer, used)
        }
      }

      // Try to read one more byte to see whether we have just reached EOF.
      n = stream.read()
      if (n < 0) {
        if (used == buffer.length) {
          return buffer
        }
        else {
          return java.util.Arrays.copyOf(buffer, used)
        }
      }

      // Well.. that worked, reallocate buffer before we try again.
      val newSizeHint = Math.max(8192L, used * 2L)
      if (newSizeHint > ArrayEx.maxSize) {
        throw new OutOfMemoryError("Larger than maximum array size.")
      }
      buffer = java.util.Arrays.copyOf(buffer, newSizeHint.toInt)

      // Add the byte to the buffer.
      buffer(used) = n.toByte
      used += 1
    }

    throw new UnknownError
  }

  @inline
  final def readLines(stream:  InputStream,
                      charset: Charset = StandardCharsets.US_ASCII)
  : Array[String] = {
    val builder = Array.newBuilder[String]
    val reader = new BufferedReader(new InputStreamReader(stream, charset.newDecoder()))
    var line = reader.readLine()
    while (line != null) {
      builder += line
      line = reader.readLine()
    }
    builder.result()
  }

  @inline
  final def readText(stream:  InputStream,
                     charset: Charset = StandardCharsets.US_ASCII)
  : String = {
    val builder = StringBuilder.newBuilder
    val reader = new BufferedReader(new InputStreamReader(stream, charset.newDecoder()))
    var line = reader.readLine()
    while (line != null) {
      builder ++= line
      line = reader.readLine()
    }
    builder.result()
  }

  @inline
  final def writeLines(stream:  OutputStream,
                       lines:   Array[String],
                       charset: Charset = StandardCharsets.US_ASCII)
  : Unit = {
    val writer = new BufferedWriter(new OutputStreamWriter(stream, charset.newEncoder()))
    ArrayEx.foreach(lines)(line => {
      writer.write(line)
      writer.newLine()
    })
    writer.flush()
  }

  @inline
  final def writeText(stream:  OutputStream,
                      text:    String,
                      charset: Charset = StandardCharsets.US_ASCII)
  : Unit = {
    val writer = new BufferedWriter(new OutputStreamWriter(stream, charset.newEncoder()))
    writer.write(text)
    writer.flush()
  }

  @inline
  final def writeJson(stream: OutputStream,
                      json:   JValue,
                      pretty: Boolean = true)
  : Unit = {
    val mapper = JsonMethods.mapper
    mapper.configure(SerializationFeature.CLOSE_CLOSEABLE, false)
    // Not sure why mapper still closes the stream, even with the flag being set.
    // So let's just create dummy stream around it to avoid this issue.
    val dummyStream = new OutputStream {

      override def close()
      : Unit = {
        //super.close()
      }

      override def flush()
      : Unit = {
        //super.flush()
        stream.flush()
      }

      override def write(b: Int)
      : Unit = stream.write(b)

      override def write(b: Array[Byte])
      : Unit = {
        //super.write(b)
        stream.write(b)
      }

      override def write(b: Array[Byte], off: Int, len: Int)
      : Unit = {
        // super.write(b, off, len)
        stream.write(b, off, len)
      }

    }

    if (pretty) {
      val writer = mapper.writerWithDefaultPrettyPrinter()
      writer.writeValue(dummyStream, json)
    }
    else {
      mapper.writeValue(dummyStream, json)
    }
  }

}
