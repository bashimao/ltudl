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

package object image {

  final val LTU_IO_IMAGE_DEFAULT_IMPLEMENTATION
  : String = {
    val tmp = Environment.get(
      "LTU_IO_IMAGE_DEFAULT_IMPLEMENTATION",
      "AWT",
      _.length > 0
    )
    tmp
  }

  final val Bitmap
  : BitmapBuilder = LTU_IO_IMAGE_DEFAULT_IMPLEMENTATION match {
    case "AWT" =>
      AWTBitmap
    case "ImageMagick" =>
      ImageMagickBitmap
    case "OpenCV" =>
      OpenCVBitmap
    case _ =>
      throw new MatchError(LTU_IO_IMAGE_DEFAULT_IMPLEMENTATION)
  }

}
