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
import it.unimi.dsi.fastutil.io._
import java.awt._
import java.awt.geom._
import java.awt.image._
import java.io._
import java.nio._
import java.nio.channels._
import javax.imageio._
import scala.collection._

/**
  * Some functions that make handling images easier.
  */
final class AWTBitmap(private var image: BufferedImage)
  extends BitmapEx[AWTBitmap]
    with AutoClosing {
  require(image != null)

  override def toString
  : String = s"AWTBitmap[${image.getWidth} x ${image.getHeight} x $format]"

  override protected def doClose()
  : Unit = {
    image.flush()
    image = null
    super.doClose()
  }

  def copy
  : AWTBitmap = {
    val result = new BufferedImage(
      image.getWidth, image.getHeight, image.getType
    )
    val g = result.getGraphics
    g.drawImage(image, 0, 0, null)
    g.dispose()
    AWTBitmap(result)
  }

  override protected def doCrop(x: Int, y: Int, w: Int, h: Int)
  : AWTBitmap = {
    val result = image.getSubimage(x, y, w, h)
    AWTBitmap(result)
  }

  override def flipHorizontal()
  : AWTBitmap = {
    // Create affine transformation.
    val s = AffineTransform.getScaleInstance(-1.0, 1.0)
    val t = AffineTransform.getTranslateInstance(-width, 0.0)
    val at = new AffineTransform()
    at.concatenate(s)
    at.concatenate(t)

    // Apply affine transformation.
    val ato = new AffineTransformOp(at, AffineTransformOp.TYPE_NEAREST_NEIGHBOR)
    AWTBitmap(ato.filter(image, null))
  }

  override def flipVertical()
  : AWTBitmap = {
    // Create affine transformation.
    val s = AffineTransform.getScaleInstance(1.0, -1.0)
    val t = AffineTransform.getTranslateInstance(0.0, -height)
    val at = new AffineTransform()
    at.concatenate(s)
    at.concatenate(t)

    // Apply affine transformation.
    val ato = new AffineTransformOp(at, AffineTransformOp.TYPE_NEAREST_NEIGHBOR)
    AWTBitmap(ato.filter(image, null))
  }

  override def format
  : BitmapFormat = image.getType match {
    case BufferedImage.TYPE_BYTE_GRAY =>
      BitmapFormat.Grayscale
    case BufferedImage.TYPE_3BYTE_BGR =>
      BitmapFormat.BGR
    case BufferedImage.TYPE_4BYTE_ABGR =>
      BitmapFormat.BGRWithAlpha
    case _ =>
      throw new MatchError(image.getType)
  }

  override def height
  : Int = image.getHeight

  override def noChannels
  : Int = image.getType match {
    case BufferedImage.TYPE_BYTE_GRAY =>
      1
    case BufferedImage.TYPE_3BYTE_BGR =>
      3
    case BufferedImage.TYPE_4BYTE_ABGR =>
      4
    case _ =>
      throw new MatchError(format)
  }

  override def resample(width: Int, height: Int, format: BitmapFormat)
  : AWTBitmap = {
    val typeID = format match {
      case BitmapFormat.Grayscale =>
        BufferedImage.TYPE_BYTE_GRAY
      case BitmapFormat.BGR =>
        BufferedImage.TYPE_3BYTE_BGR
      case BitmapFormat.BGRWithAlpha =>
        BufferedImage.TYPE_4BYTE_ABGR
      case _ =>
        throw new MatchError(format)
    }
    val result = new BufferedImage(width, height, typeID)
    val g = result.createGraphics
    g.setRenderingHint(
      RenderingHints.KEY_INTERPOLATION,
      RenderingHints.VALUE_INTERPOLATION_BICUBIC
    )
    g.drawImage(image, 0, 0, width, height, null)
    g.dispose()
    AWTBitmap(result)
  }

  override def encode(encoding: String)
  : Array[Byte] = {
    using(new FastByteArrayOutputStream(1024 * 1024))(stream => {
      ImageIO.write(image, encoding, stream)
      stream.trim()
      stream.array
    })
  }

  override def toByteArray
  : Array[Byte] = {
    val db = image.getRaster.getDataBuffer
    val n  = size

    // Make sure we can access indexes directly.
    if (n != db.getSize) {
      return using(
        copy
      )(_.toByteArray)
    }

    // Copy pixel data into array. (Does implicit BGR to RGB conversion).
    val result = new Array[Byte](n)
    var i = 0
    while (i < result.length) {
      result(i) = db.getElem(i).toByte
      i += 1
    }
    result
  }

  override def toDoubleArray
  : Array[Double] = {
    val db = image.getRaster.getDataBuffer
    val n  = size

    // Make sure we can access indexes directly.
    if (n != db.getSize) {
      return using(
        copy
      )(_.toDoubleArray)
    }

    // Copy pixel data into array. (Does implicit BGR to RGB conversion).
    val result      = new Array[Double](n)
    val maxValueInv = 1.0 / 255.0
    var i           = 0
    while (i < result.length) {
      result(i) = db.getElem(i) * maxValueInv
      i += 1
    }
    result
  }

  override def toFloatArray
  : Array[Float] = {
    val db = image.getRaster.getDataBuffer
    val n  = size

    // Make sure we can access indexes directly.
    if (n != db.getSize) {
      return using(
        copy
      )(_.toFloatArray)
    }

    // Copy pixel data into array. (Does implicit BGR to RGB conversion).
    val result      = new Array[Float](n)
    val maxValueInv = 1.0f / 255.0f
    var i           = 0
    while (i < result.length) {
      result(i) = db.getElem(i) * maxValueInv
      i += 1
    }
    result
  }

  override def toIntArray
  : Array[Int] = {
    val db = image.getRaster.getDataBuffer
    val n  = size

    // Make sure we can access indexes directly.
    if (n != db.getSize) {
      return using(
        copy
      )(_.toIntArray)
    }

    val result = new Array[Int](size)
    var i      = 0
    while (i < result.length) {
      result(i) = db.getElem(i)
      i += 1
    }
    result
  }

  override def width
  : Int = image.getWidth

  def ++(other: AWTBitmap)
  : AWTBitmap = {
    require(width == other.width)

    val height0 = height
    val height1 = height + other.height

    val result = new BufferedImage(width, height1, image.getType)
    val g = result.getGraphics
    g.drawImage(image,       0, 0,       null)
    g.drawImage(other.image, 0, height0, null)
    g.dispose()
    AWTBitmap(result)
  }

  /*
  @inline
  def boundingBox(rgbKey: Int): (Range, Range) = {
    var x0 = Int.MaxValue
    var x1 = Int.MinValue
    var y0 = Int.MaxValue
    var y1 = Int.MinValue
    cfor(0)(_ < bmp.getHeight, _ + 1)(y => {
      cfor(0)(_ < bmp.getWidth, _ + 1)(x => {
        if (bmp.getRGB(x, y) != rgbKey) {
          x0 = Math.min(x0, x)
          x1 = Math.max(x1, x)
          y0 = Math.min(y0, y)
          y1 = Math.max(y1, y)
        }
      })
    })
    if (x0 > x1 || y0 > y1) {
      null
    }
    else {
      (x0 until x1, y0 until y1)
    }
  }
  */

  /*
  @inline
  def extractActivationsLuma709: DVec = {
    val width  = bmp.getWidth
    val height = bmp.getHeight
    val result = new Array[Real](width * height)
    var i = 0
    var y = 0
    while (y < height) {
      var x = 0
      while (x < width) {
        // Get pixel
        val rgb = bmp.getRGB(x, y)
        x += 1
        // Extract RGB values
        val r = ((rgb >> 16) & 0xFF) * luminanceFactorR
        val g = ((rgb >>  8) & 0xFF) * luminanceFactorG
        val b = ( rgb        & 0xFF) * luminanceFactor
        result(i) = r + g + b
        i += 1
      }
      y += 1
    }
    DVec(result)
  }

/*

Perceived brightness as described by Darel Rex Finley in (HSP Color Model, 2006)
 */
  @inline
  def extractActivationsLumaHSP: DVec = {
    val width  = bmp.getWidth
    val height = bmp.getHeight
    val result = new Array[Real](width * height)
    var i = 0
    var y = 0
    while (y < height) {
      var x = 0
      while (x < width) {
        // Get pixel
        val rgb = bmp.getRGB(x, y)
        x += 1
        // Extract RGB values
        val r = ((rgb >> 16) & 0xFF) * luminanceFactorR
        val g = ((rgb >>  8) & 0xFF) * luminanceFactorG
        val b = ( rgb        & 0xFF) * luminanceFactor
        result(i) = sqrt(r*r + g*g + b*b)
        i += 1
      }
      y += 1
    }
    DVec(result)
  }
  */

}

object AWTBitmap
  extends BitmapBuilder {

  ImageIO.setUseCache(false)

  override def toString
  : String = "AWTBitmapBuilder"

  final private def apply(image: BufferedImage)
  : AWTBitmap = new AWTBitmap(image)

  final override def apply(width:  Int,
                           height: Int,
                           format: BitmapFormat)
  : AWTBitmap = {
    val _type = format match {
      case BitmapFormat.Grayscale =>
        BufferedImage.TYPE_BYTE_GRAY

      case BitmapFormat.BGR =>
        BufferedImage.TYPE_3BYTE_BGR

      case BitmapFormat.BGRWithAlpha =>
        BufferedImage.TYPE_4BYTE_ABGR

      case _ =>
        throw new MatchError(format)
    }
    val image = new BufferedImage(width, height, _type)
    apply(image)
  }


  final override def derive(width:  Int,
                            height: Int,
                            format: BitmapFormat,
                            pixels: Array[Byte])
  : AWTBitmap = {
    val result = apply(width, height, format)
    val db     = result.image.getRaster.getDataBuffer
    var i      = 0
    val n      = result.size
    while (i < n) {
      db.setElem(i, MathMacros.toUnsigned(pixels(i)))
      i += 1
    }
    result
  }

  final def derive(width:  Int,
                   height: Int,
                   format: BitmapFormat,
                   pixels: Array[Int])
  : AWTBitmap = {
    val result = apply(width, height, format)
    val db     = result.image.getRaster.getDataBuffer
    var i      = 0
    val n      = result.size
    while (i < n) {
      db.setElem(i, pixels(i))
      i += 1
    }
    result
  }

  final override def derive(width:  Int,
                            height: Int,
                            format: BitmapFormat,
                            pixels: Array[Float])
  : AWTBitmap = {
    val result = apply(width, height, format)
    val db     = result.image.getRaster.getDataBuffer
    var i      = 0
    val n      = result.size
    while (i < n) {
      db.setElem(i, (pixels(i) * 255.0f + 0.5f).toInt)
      i += 1
    }
    result
  }

  final override def derive(width:  Int,
                            height: Int,
                            format: BitmapFormat,
                            pixels: Array[Double])
  : AWTBitmap = {
    val result = apply(width, height, format)
    val db     = result.image.getRaster.getDataBuffer
    var i      = 0
    val n      = result.size
    while (i < n) {
      db.setElem(i, (pixels(i) * 255.0 + 0.5).toInt)
      i += 1
    }
    result
  }

  final override def decode(array: Array[Byte])
  : AWTBitmap = decode(ArrayEx.toInputStream(array))

  final override def decode(buffer: ByteBuffer)
  : AWTBitmap = using(ByteBufferBackedInputStream(buffer))(decode)

  final override def decode(stream: InputStream)
  : AWTBitmap = {
    var tries = 0
    do {
      try {
        val bmp = ImageIO.read(stream)
        return apply(bmp)
      }
      catch {
        case e: OutOfMemoryError =>
          logger.error(s"Exception caught: ", e)
          System.gc()
          System.runFinalization()
      }
      tries += 1
    } while (tries < 100)
    // TODO: Make this a configurable!
    throw new UnknownError
  }

  final override def decode(channel: ReadableByteChannel)
  : AWTBitmap = decode(Channels.newInputStream(channel))

  final override def decode(file: FileHandle)
  : AWTBitmap = {
    using(
      file.openStream()
    )(decode)
  }

  /**
   * @param fontName Java font name. Universal font names:
   *                 Font.SERIF,
   *                 Font.SANS_SERIF,
   *                 Font.MONOSPACED,
   *                 Font.DIALOG,
   *                 Font.DIALOG_INPUT
   * @return
   */
  final def visualizeNeurons(layout:       (Int, Int),
                             displayWidth: Int,
                             neurons:      SortedMap[Long, Array[Real]],
                             bgColor:      Color   = Color.magenta,
                             fontName:     String  = Font.SANS_SERIF,
                             fontSize:     Int     = 8,
                             fontColor:    Color   = Color.white,
                             tileSpacingX: Int     = 3,
                             tileSpacingY: Int     = 3,
                             omitWeight0:  Boolean = true)
  : AWTBitmap = {
    require(tileSpacingX >= 0)
    require(tileSpacingY >= 0)

    val displayHeight = {
      (neurons.map(_._2.length).max + displayWidth - 1) / displayWidth
    }

    // Create font and get actual height.
    val font = new Font(fontName, java.awt.Font.PLAIN, fontSize)
    val (baselineOffset, labelHeight) = {
      val fm = new BufferedImage(
        1, 1, BufferedImage.TYPE_INT_RGB
      ).getGraphics.getFontMetrics(font)
      (fm.getAscent, fm.getAscent + Math.max(1, fm.getDescent / 2))
    }

    // Compute image size.
    val tileStrideX = tileSpacingX + displayWidth
    val tileStrideY = tileSpacingY + displayHeight + labelHeight
    val width       = tileSpacingX + layout._1 * tileStrideX
    val height      = tileSpacingY + layout._2 * tileStrideY

    // Allocate memory for rendering image.
    val image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)
    val g = image.createGraphics()
    g.setFont(font)

    // Draw background.
    g.setPaint(bgColor)
    g.fillRect(0, 0, width, height)

    // Find min and max value for scaling.
    val globalMin = MapEx.foldLeftValues(Real.positiveInfinity, neurons)(
      (a, b) => Math.min(a, ArrayEx.min(b))
    )
    val globalMax = MapEx.foldLeftValues(Real.negativeInfinity, neurons)(
      (a, b) => Math.max(a, ArrayEx.max(b))
    )

    // Draw neurons.
    val neuronIter = neurons.iterator
    var y0 = tileStrideY
    while (y0 < height) {
      var x0 = tileSpacingX
      while (x0 < width) {
        if (neuronIter.hasNext) {
          // Normalize weights.
          val (neuronNo, weightsIn) = neuronIter.next()
          val weights: Array[Real] = weightsIn.clone
          //weights :-= min(weights)
          //weights :/= max(weights)
          ArrayEx.add(weights, -globalMin)
          ArrayEx.multiply(weights, Real.one / (globalMax - globalMin))

          // Draw title.
          g.setPaint(fontColor)
          g.drawString(neuronNo.toString, x0, y0 + baselineOffset)

          // Draw pixels.
          val yFirstPixel = y0 + labelHeight

          var i = 0
          var j = 0
          ArrayEx.foreach(weights)(weight => {
            val r   = (weight * 255.0 + 0.5).toInt
            val bgr = (r * 256 * 256) + (r * 256) + r
            image.setRGB(x0 + i, yFirstPixel + j, bgr)
            i = (i + 1) % displayWidth
            if (i == 0) {
              j = (j + 1) % displayHeight
            }
          })
        }

        x0 += tileStrideX
      }

      y0 += tileStrideY
    }
    g.dispose()

    // Return image.
    apply(image)
  }

}
