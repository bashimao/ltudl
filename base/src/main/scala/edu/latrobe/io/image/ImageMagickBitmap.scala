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

import java.awt._
import java.io._
import java.nio._
import java.nio.channels.{Channels, ReadableByteChannel, WritableByteChannel}

import edu.latrobe._
import edu.latrobe.io._
import magick._

final class ImageMagickBitmap(private val image: MagickImage)
  extends BitmapEx[ImageMagickBitmap]
    with AutoClosing {
  require(image != null)

  override def toString
  : String = {
    val dims = image.getDimension
    s"ImageMagickBitmap[${dims.width} x ${dims.height} x $format]"
  }

  override protected def doClose()
  : Unit = {
    image.destroyImages()
    super.doClose()
  }

  override def copy
  : ImageMagickBitmap = {
    val cloned = image.cloneImage(0, 0, false)
    ImageMagickBitmap(cloned)
  }

  override protected def doCrop(x: Int, y: Int, w: Int, h: Int)
  : ImageMagickBitmap = {
    val cropped = image.cropImage(new Rectangle(x, y, w, h))
    ImageMagickBitmap(cropped)
  }

  override def flipHorizontal()
  : ImageMagickBitmap = {
    val flipped = image.flopImage()
    ImageMagickBitmap(flipped)
  }

  override def flipVertical()
  : ImageMagickBitmap = {
    val flipped = image.flipImage()
    ImageMagickBitmap(flipped)
  }

  override def format
  : BitmapFormat = image.getColorspace match {
    case ColorspaceType.GRAYColorspace =>
      BitmapFormat.Grayscale
    case _ =>
      BitmapFormat.BGR
  }

  override def height
  : Int = image.getDimension.height

  override def noChannels
  : Int = image.getColorspace match {
    case ColorspaceType.GRAYColorspace =>
      1
    case _ =>
      3
  }

  override def resample(width:  Int,
                        height: Int,
                        format: BitmapFormat)
  : ImageMagickBitmap = synchronized {
    // Because setFilter might not be thread-safe!
    image.setFilter(FilterType.CubicFilter)
    val zoomed = image.zoomImage(width, height)

    // Chance colorspace if necessary!
    format match {
      case BitmapFormat.Grayscale =>
        if (!zoomed.isGrayImage) {
          val success = zoomed.rgbTransformImage(ColorspaceType.GRAYColorspace)
          assume(success)
        }

      case BitmapFormat.BGR =>
        if (zoomed.isGrayImage) {
          val success = zoomed.transformRgbImage(ColorspaceType.sRGBColorspace)
          assume(success)
        }

      case _ =>
        throw new MatchError(format)
    }

    ImageMagickBitmap(zoomed)
  }

  override def encode(encoding: String)
  : Array[Byte] = {
    val info = new ImageInfo()
    info.setMagick(encoding)
    image.imageToBlob(info)
  }

  override def toByteArray
  : Array[Byte] = {
    val dims    = image.getDimension
    val result  = new Array[Byte](dims.width * dims.height * noChannels)
    val success = image.dispatchImage(
      0, 0, dims.width, dims.height,
      ImageMagickBitmap.formatStringFor(format),
      result
    )
    assume(success)
    result
  }

  override def toIntArray
  : Array[Int] = {
    val bytes = toByteArray
    ArrayEx.map(bytes)(MathMacros.toUnsigned(_))
  }

  override def toDoubleArray
  : Array[Double] = {
    ArrayEx.map(
      toFloatArray
    )(_.toDouble)
  }

  override def toFloatArray
  : Array[Float] = {
    val dims    = image.getDimension
    val result  = new Array[Float](dims.width * dims.height * noChannels)
    val success = image.dispatchImage(
      0, 0, dims.width, dims.height,
      ImageMagickBitmap.formatStringFor(format),
      result
    )
    assume(success)
    result
  }

  override def width
  : Int = image.getDimension.width

}

object ImageMagickBitmap
  extends BitmapBuilder {

  override def toString
  : String = "ImageMagickBitmapBuilder"

  final private def apply(image: MagickImage)
  : ImageMagickBitmap = new ImageMagickBitmap(image)


  final override def apply(width:  Int,
                           height: Int,
                           format: BitmapFormat)
  : ImageMagickBitmap = {
    val tmp = new Array[Byte](width * height * format.noChannels)
    derive(width, height, format, tmp)
  }

  final override def derive(width:  Int,
                            height: Int,
                            format: BitmapFormat,
                            pixels: Array[Byte])
  : ImageMagickBitmap = {
    val image = new MagickImage()
    image.constituteImage(
      width, height, formatStringFor(format),
      pixels
    )
    apply(image)
  }

  final def derive(width:  Int,
                   height: Int,
                   format: BitmapFormat,
                   pixels: Array[Int])
  : ImageMagickBitmap = {
    val image = new MagickImage()
    image.constituteImage(
      width, height, formatStringFor(format),
      pixels
    )
    apply(image)
  }

  final override def derive(width:  Int,
                            height: Int,
                            format: BitmapFormat,
                            pixels: Array[Float])
  : ImageMagickBitmap = {
    val image = new MagickImage()
    image.constituteImage(
      width, height, formatStringFor(format),
      pixels
    )
    apply(image)
  }

  final override def derive(width:  Int,
                            height: Int,
                            format: BitmapFormat,
                            pixels: Array[Double])
  : Bitmap = derive(
    width,
    height,
    format,
    ArrayEx.map(pixels)(_.toFloat)
  )

  final private def formatStringFor(format: BitmapFormat)
  : String = format match {
    case BitmapFormat.Grayscale =>
      "I"
    case BitmapFormat.BGR =>
      "BGR"
    case BitmapFormat.BGRWithAlpha =>
      "BGRA"
    case _ =>
      throw new MatchError(format)
  }

  final override def decode(array: Array[Byte])
  : ImageMagickBitmap = {
    val image = new MagickImage()
    image.blobToImage(new ImageInfo(), array)
    apply(image)
  }

  final override def decode(buffer: ByteBuffer)
  : ImageMagickBitmap = {
    using(
      ByteBufferBackedInputStream(buffer)
    )(decode)
  }

  final override def decode(stream: InputStream)
  : ImageMagickBitmap = decode(StreamEx.read(stream, 1024 * 1024))

  final override def decode(channel: ReadableByteChannel)
  : ImageMagickBitmap = decode(Channels.newInputStream(channel))

  final override def decode(file: FileHandle)
  : ImageMagickBitmap = decode(file.readAsArray())

}
