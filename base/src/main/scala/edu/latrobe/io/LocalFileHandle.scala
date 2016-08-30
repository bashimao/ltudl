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
import java.net._
import java.nio._
import java.nio.channels._
import java.nio.file._
import org.apache.commons.io._
import org.json4s.JsonAST._
import scala.collection._
import scala.util._
import scala.util.hashing._

@SerialVersionUID(1L)
final class LocalFileHandle(val path: String)
  extends FileHandle {

  override def toString
  : String = path

  override def hashCode()
  : Int = MurmurHash3.mix(super.hashCode(), path.hashCode())

  override def canEqual(that: Any)
  : Boolean = that.isInstanceOf[LocalFileHandle]

  override protected def doEquals(other: Equatable)
  : Boolean = super.doEquals(other) && (other match {
    case other: LocalFileHandle =>
      path == other.path
    case _ =>
      false
  })

  override def isDirectory
  : Boolean = Files.isDirectory(toPath, LinkOption.NOFOLLOW_LINKS)

  /**
    * Extends the path name.
    */
  override def ++(namePart: String)
  : LocalFileHandle = LocalFileHandle(FilenameUtils.concat(path, namePart))

  override def withoutExtension()
  : LocalFileHandle = LocalFileHandle(FilenameUtils.removeExtension(path))


  // ---------------------------------------------------------------------------
  //    Read/Write
  // ---------------------------------------------------------------------------
  override def createStream(append: Boolean)
  : FileOutputStream = new FileOutputStream(path, append)

  override def createChannel(append: Boolean)
  : FileChannel = {
    if (append) {
      FileChannel.open(
        toPath,
        StandardOpenOption.WRITE,
        StandardOpenOption.CREATE
      )
    }
    else {
      FileChannel.open(
        toPath,
        StandardOpenOption.WRITE,
        StandardOpenOption.CREATE,
        StandardOpenOption.TRUNCATE_EXISTING
      )
    }
  }

  override def openStream()
  : FileInputStream = new FileInputStream(path)

  override def openChannel()
  : FileChannel = FileChannel.open(toPath, StandardOpenOption.READ)

  override def readAsArray()
  : Array[Byte] = Files.readAllBytes(toPath)

  override def readAsBuffer()
  : ByteBuffer = {
    using(openChannel())(channel => {
      val size = channel.size()
      assume(size <= ArrayEx.maxSize)

      var tries = 0
      do {
        try {
          val result = ByteBuffer.allocateDirect(size.toInt)
          while (result.hasRemaining) {
            channel.read(result)
          }
          result.rewind()
          return result
        }
        catch {
          case e: OutOfMemoryError =>
            logger.error(s"Exception caught: ", e)
            System.gc()
            System.runFinalization()
        }
        tries += 1
      } while (tries < 100)
      throw new OutOfMemoryError("Unable to recover from out of memory situation!")
    })
  }


  // ---------------------------------------------------------------------------
  //    Conversion / more information.
  // ---------------------------------------------------------------------------
  def toFile
  : File = new File(path)

  def toPath
  : Path = Paths.get(path)

  def toURI
  : URI = new URI("file", "localhost", path)

  def toURL
  : URL = new URL("file", "localhost", path)

  override protected def doToJson()
  : List[JField] = List(
    Json.field("path", path)
  )


  // ---------------------------------------------------------------------------
  //    Directory services.
  // ---------------------------------------------------------------------------
  override protected def doTraverse(depth:       Int,
                                    onDirectory: (Int, FileHandle) => Boolean,
                                    onFile:      (Int, FileHandle) => Unit)
  : Unit = {
    if (onDirectory(depth, this)) {
      using(Files.newDirectoryStream(Paths.get(path)))(stream => {
        val iter = stream.iterator()

        while (iter.hasNext) {
          val path  = iter.next()
          val isDir = Files.isDirectory(path, LinkOption.NOFOLLOW_LINKS)
          try {
            val handle = LocalFileHandle(path.toString)
            if (isDir) {
              handle.doTraverse(depth + 1, onDirectory, onFile)
            }
            else {
              onFile(depth + 1, handle)
            }
          }
          catch {
            case e: Exception =>
              logger.error(s"LocalFileHandle.traverse[$path] => $e")
              if (LTU_IO_FILE_HANDLE_RETHROW_EXCEPTIONS_DURING_TRAVERSE) {
                throw e
              }
          }
        }
      })
    }
  }

}

object LocalFileHandle
  extends JsonSerializableCompanionEx[LocalFileHandle] {

  @inline
  final def apply(path: String)
  : LocalFileHandle = new LocalFileHandle(path)

  final override def derive(fields: Map[String, JValue])
  : LocalFileHandle = apply(
    Json.toString(fields("path"))
  )

  final val empty
  : LocalFileHandle = apply("")

  final val root
  : LocalFileHandle = apply(File.separator)

  final val userHome
  : LocalFileHandle = apply(Properties.userHome)

  final val tmp
  : LocalFileHandle = apply(Properties.tmpDir)

  @inline
  final def workingDirectory
  : LocalFileHandle = apply(new File(".").toString)

}


//object LocalPath {

  /*
  final private val fs
  : FileSystem = FileSystems.getDefault

  @inline
  final def apply(path: String)
  : LocalPath = fs.getPath(path)

  @inline
  final def apply(file: File)
  : LocalPath = apply(file.getPath)

  @inline
  final def apply(uri: URI)
  : LocalPath = apply(uri.getPath)

  @inline
  final def concat(path0: LocalPath, str1: String)
  : LocalPath = fs.getPath(path0.toString, str1)

  @inline
  final def concat(path0: LocalPath, str1: String, str2: String)
  : LocalPath = fs.getPath(path0.toString, str1, str2)

  @inline
  final def concat(path0: LocalPath, str1: String, str2: String, str3: String)
  : LocalPath = fs.getPath(path0.toString, str1, str2, str3)

  @inline
  final def concat(path0: LocalPath, str1: String*)
  : LocalPath = Paths.get(path0.toString, str1(0), str1(0))

  @inline
  final def concat(path0: LocalPath, path1: LocalPath)
  : LocalPath = Paths.get(path0.toString, path1.toString)
  */

//}
