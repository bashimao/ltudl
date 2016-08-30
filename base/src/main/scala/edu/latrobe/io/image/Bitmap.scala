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

import edu.latrobe._
import edu.latrobe.io._
import java.io._
import java.nio._
import java.nio.channels._
import org.json4s.JsonAST._

abstract class Bitmap
  extends ClosableEx
    with CopyableEx[Bitmap]
    with JsonSerializable {

  def crop(xy: (Int, Int), dims: (Int, Int))
  : Bitmap

  /**
    * @param x If negative, then relative to right.
    * @param y If negative, then relative to bottom.
    * @param w If negative, then centered.
    * @param h If negative, then centered.
    * @return
    */
  def crop(x: Int, y: Int, w: Int, h: Int)
  : Bitmap

  def cropCenter(w: Int, h: Int)
  : Bitmap

  def cropCenterSquare()
  : Bitmap

  final def dims
  : (Int, Int) = (width, height)

  def flipHorizontal()
  : Bitmap

  def flipVertical()
  : Bitmap

  def format
  : BitmapFormat

  def height
  : Int

  def noChannels
  : Int

  def resample(dims: (Int, Int), format: BitmapFormat)
  : Bitmap

  def resample(w: Int, h: Int, format: BitmapFormat)
  : Bitmap

  final def encode()
  : Array[Byte] = encode("PNG")

  def encode(encoding: String)
  : Array[Byte]

  final def encode(stream: OutputStream)
  : Unit = encode(stream, "PNG")

  final def encode(stream: OutputStream, encoding: String)
  : Unit = stream.write(encode(encoding))

  final def encode(channel: WritableByteChannel)
  : Unit = encode(channel, "PNG")

  final def encode(channel: WritableByteChannel, encoding: String)
  : Unit = encode(Channels.newOutputStream(channel), encoding)

  final def encode(file: FileHandle)
  : Unit = encode(file, "PNG")

  final def encode(file: FileHandle, encoding: String)
  : Unit = file.write(encode(encoding))

  final def size: Int = width * height * noChannels

  def toByteArray
  : Array[Byte]

  def toDoubleArray
  : Array[Double]

  def toFloatArray
  : Array[Float]

  def toIntArray
  : Array[Int]

  final def toRealArray
  : Array[Real] = {
    // -------------------------------------------------------------------------
    //    REAL SWITCH DOUBLE
    // -------------------------------------------------------------------------
    /*
    toDoubleArray
    */
    // -------------------------------------------------------------------------
    //    REAL SWITCH FLOAT
    // -------------------------------------------------------------------------
    ///*
    toFloatArray
    //*/
    // -------------------------------------------------------------------------
    //    REAL SWITCH END
    // -------------------------------------------------------------------------
  }

  def width
  : Int

  final override protected def doToJson()
  : List[JField] = List(
    Json.field("width",  width),
    Json.field("height", height),
    Json.field("data",   toByteArray)
  )

}

abstract class BitmapEx[TThis <: BitmapEx[_]]
  extends Bitmap {

  override def copy: TThis

  final override def crop(xy: (Int, Int), dims: (Int, Int))
  : TThis = crop(xy._1, xy._2, dims._1, dims._2)

  final override def crop(x: Int, y: Int, w: Int, h: Int)
  : TThis = {
    var x0 = x
    if (x < 0) {
      x0 += width
    }
    var w0 = w
    if (w < 0) {
      x0 += w / 2
      w0 = -w
    }
    var y0 = y
    if (y < 0) {
      y0 += height
    }
    var h0 = h
    if (h < 0) {
      y0 += h / 2
      h0 = -h
    }
    doCrop(x0, y0, w0, h0)
  }

  final override def cropCenter(w: Int, h: Int)
  : TThis = crop(width / 2, height / 2, -w, -h)

  final override def cropCenterSquare()
  : TThis = {
    val minDim = Math.min(width, height)
    cropCenter(minDim, minDim)
  }

  protected def doCrop(x: Int, y: Int, w: Int, h: Int)
  : TThis

  override def flipHorizontal()
  : TThis

  override def flipVertical()
  : TThis

  final override def resample(dims: (Int, Int), format: BitmapFormat)
  : TThis = resample(dims._1, dims._2, format)

  override def resample(width: Int, height: Int, format: BitmapFormat)
  : TThis

}

abstract class BitmapBuilder {

  def apply(width:  Int,
            height: Int,
            format: BitmapFormat)
  : Bitmap

  def derive(width:  Int,
             height: Int,
             format: BitmapFormat,
             pixels: Array[Byte])
  : Bitmap

  def derive(width:  Int,
             height: Int,
             format: BitmapFormat,
             pixels: Array[Float])
  : Bitmap

  def derive(width:  Int,
             height: Int,
             format: BitmapFormat,
             pixels: Array[Double])
  : Bitmap

  final def derive(dims:   (Int, Int),
                   format: BitmapFormat,
                   pixels: Array[Byte])
  : Bitmap = derive(dims._1, dims._2, format, pixels)

  final def derive(dims:   (Int, Int),
                   format: BitmapFormat,
                   pixels: Array[Float])
  : Bitmap = derive(dims._1, dims._2, format, pixels)

  final def derive(dims:   (Int, Int),
                   format: BitmapFormat,
                   pixels: Array[Double])
  : Bitmap = derive(dims._1, dims._2, format, pixels)

  def decode(array: Array[Byte])
  : Bitmap

  def decode(buffer: ByteBuffer)
  : Bitmap

  def decode(stream: InputStream)
  : Bitmap

  def decode(channel: ReadableByteChannel)
  : Bitmap

  def decode(file: FileHandle)
  : Bitmap

}
