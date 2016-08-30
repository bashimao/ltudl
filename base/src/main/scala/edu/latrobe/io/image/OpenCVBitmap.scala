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

package edu.latrobe.io.image

import java.io._
import java.nio._
import java.nio.IntBuffer
import java.nio.channels._

import edu.latrobe._
import edu.latrobe.io._
import edu.latrobe.native._
import org.bytedeco.javacpp._
import org.bytedeco.javacpp.indexer._
import org.bytedeco.javacpp.opencv_core._
import org.bytedeco.javacpp.opencv_imgcodecs._
import org.bytedeco.javacpp.opencv_imgproc._

final class OpenCVBitmap(private var mat: Mat)
  extends BitmapEx[OpenCVBitmap]
    with AutoClosing {
  require(mat != null)

  override def toString
  : String = s"OpenCVBitmap[${mat.cols} x ${mat.rows} x $format]"

  override protected def doClose()
  : Unit = {
    mat.release()
    super.doClose()
  }

  override def copy
  : OpenCVBitmap = OpenCVBitmap(mat.clone())

  override protected def doCrop(x: Int, y: Int, w: Int, h: Int)
  : OpenCVBitmap = {
    using(new Rect(x, y, w, h))(r => {
      OpenCVBitmap(
        mat(r)
      )
    })
  }

  override def flipHorizontal()
  : OpenCVBitmap = {
    val result = OpenCVBitmap(mat.cols, mat.rows, format)
    flip(mat, result.mat, 1)
    result
  }

  override def flipVertical()
  : OpenCVBitmap = {
    val result = OpenCVBitmap(mat.cols, mat.rows, format)
    flip(mat, result.mat, 0)
    result
  }

  override def format
  : BitmapFormat = mat.channels() match {
    case 1 =>
      BitmapFormat.Grayscale
    case 3 =>
      BitmapFormat.BGR
    case 4 =>
      BitmapFormat.BGRWithAlpha
    case _ =>
      throw new MatchError(mat.channels())
  }

  override def height
  : Int = mat.rows()

  override def noChannels
  : Int = mat.channels()

  def put(result: Array[Byte], offset: Int)
  : Int = {
    val buffer = mat.createBuffer[ByteBuffer]()
    val n      = size
    buffer.get(result, offset, n)
    offset + n
  }

  def put(result: Array[Int], offset: Int)
  : Int = {
    val buffer = mat.createBuffer[IntBuffer]()
    val n      = size
    buffer.get(result, offset, n)
    offset + n
  }

  def put(result: Array[Real], offset: Int)
  : Int = {
    val buffer = mat.createBuffer[NativeRealBuffer]()
    val n      = size
    buffer.get(result, offset, n)
    offset + n
  }

  override def resample(width:  Int,
                        height: Int,
                        format: BitmapFormat)
  : OpenCVBitmap = {
    // Resize first.
    val tmp0 = new Mat()
    using(new opencv_core.Size(width, height))(tmpSize => {
      resize(
        mat,
        tmp0,
        tmpSize,
        0.0,
        0.0,
        INTER_LANCZOS4
      )
    })

    val tmp1 = tmp0.channels() match {
      case 1 =>
        format match {
          case BitmapFormat.Grayscale =>
            // do nothing.
            tmp0
          case BitmapFormat.BGR =>
            val tmp1 = new Mat()
            cvtColor(tmp0, tmp1, CV_GRAY2BGR)
            tmp0.release()
            tmp1
          case BitmapFormat.BGRWithAlpha =>
            val tmp1 = new Mat()
            cvtColor(tmp0, tmp1, CV_GRAY2BGRA)
            tmp0.release()
            tmp1
          case _ =>
            throw new MatchError(format)
        }

      case 3 =>
        format match {
          case BitmapFormat.Grayscale =>
            val tmp1 = new Mat()
            cvtColor(tmp0, tmp1, CV_BGR2GRAY)
            tmp0.release()
            tmp1
          case BitmapFormat.BGR =>
            // do nothing.
            tmp0
          case BitmapFormat.BGRWithAlpha =>
            val tmp1 = new Mat()
            cvtColor(tmp0, tmp1, CV_BGR2BGRA)
            tmp0.release()
            tmp1
          case _ =>
            throw new MatchError(format)
        }

      case 4 =>
        format match {
          case BitmapFormat.Grayscale =>
            val tmp1 = new Mat()
            cvtColor(tmp0, tmp1, CV_BGRA2GRAY)
            tmp0.release()
            tmp1
          case BitmapFormat.BGR =>
            val tmp1 = new Mat()
            cvtColor(tmp0, tmp1, CV_BGRA2BGR)
            tmp0.release()
            tmp1
          case BitmapFormat.BGRWithAlpha =>
            // do nothing.
            tmp0
          case _ =>
            throw new MatchError(format)
        }

      case _ =>
        throw new MatchError(tmp0.channels())
    }

    OpenCVBitmap(tmp1)
  }

  override def encode(encoding: String)
  : Array[Byte] = {
    using(new BytePointer())(ptr => {
      val success = imencode(s".$encoding", mat, ptr)
      assume(success)
      NativeBufferEx.toArray(ptr.asByteBuffer())
    })
  }

  override def toByteArray
  : Array[Byte] = {
    if (mat.isSubmatrix) {
      ArrayEx.map(
        toIntArray
      )(_.toByte)
    }
    else {
      val result = new Array[Byte](size)
      val buffer = mat.createBuffer[ByteBuffer]()
      buffer.get(result)
      result
    }
  }

  override def toDoubleArray
  : Array[Double] = {
    val maxValueInv = 1.0 / 255.0
    ArrayEx.map(
      toByteArray
    )(MathMacros.toUnsigned(_) * maxValueInv)
  }

  override def toFloatArray
  : Array[Real] = {
    val maxValueInv = 1.0f / 255.0f
    ArrayEx.map(
      toByteArray
    )(MathMacros.toUnsigned(_) * maxValueInv)
  }

  override def toIntArray
  : Array[Int] = {
    val w = mat.cols
    val h = mat.rows
    val c = mat.channels
    val rowStride = w * c

    val result  = new Array[Int](rowStride * h)
    val indexer = mat.createIndexer[UByteIndexer]()
    var off = 0
    var y   = 0
    while (y < h) {
      indexer.get(y, result, off, rowStride)
      off += rowStride
      y   += 1
    }
    result
  }

  override def width
  : Int = mat.cols()

}

object OpenCVBitmap
  extends BitmapBuilder {

  setNumThreads(0)

  override def toString
  : String = "OpenCVBitmapBuilder"

  final private def apply(mat: Mat)
  : OpenCVBitmap = new OpenCVBitmap(mat)

  final def apply(width:  Int,
                  height: Int,
                  format: BitmapFormat)
  : OpenCVBitmap = {
    val mat = new Mat(height, width, CV_8UC(format.noChannels))
    apply(mat)
  }


  override def derive(width:  Int,
                      height: Int,
                      format: BitmapFormat,
                      pixels: Array[Byte])
  : Bitmap = {
    val bmp    = apply(width, height, format)
    val buffer = bmp.mat.createBuffer[ByteBuffer](0)
    buffer.put(pixels)
    bmp
  }

  override def derive(width:  Int,
                      height: Int,
                      format: BitmapFormat,
                      pixels: Array[Float])
  : Bitmap = derive(
    width,
    height,
    format,
    ArrayEx.map(
      pixels
    )(x => (x * 255.0f + 0.5f).toByte)
  )

  override def derive(width:  Int,
                      height: Int,
                      format: BitmapFormat,
                      pixels: Array[Double])
  : Bitmap = derive(
    width,
    height,
    format,
    ArrayEx.map(
      pixels
    )(x => (x * 255.0 + 0.5).toByte)
  )

  override def decode(array: Array[Byte])
  : Bitmap = decode(ByteBuffer.wrap(array))

  override def decode(buffer: ByteBuffer)
  : Bitmap = {
    val mat = {
      using(
        new Mat(new BytePointer(buffer))
      )(imdecode(_, IMREAD_UNCHANGED))
    }
    apply(mat)
  }

  override def decode(stream: InputStream)
  : Bitmap = decode(StreamEx.read(stream, 1024 * 1024))

  override def decode(channel: ReadableByteChannel)
  : Bitmap = decode(ChannelEx.read(channel, 1024 * 1024))

  final override def decode(file: FileHandle)
  : Bitmap = decode(file.readAsBuffer())

}