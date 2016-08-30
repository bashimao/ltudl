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

import java.awt.Color
import scala.collection._

/**
  * The standard color palette we use when drawing graphics.
  */
object DefaultColors {

  final val black
  : Color = Color.BLACK

  final val darkGray
  : Color = Color.DARK_GRAY

  final val gray
  : Color = Color.GRAY

  final val lightGray
  : Color = Color.LIGHT_GRAY

  final val white
  : Color = Color.WHITE

  /**
    * Since we encode pictures as BGR, the first three colors should also be
    * BGR.
    *
    * Definitions from: https://en.wikipedia.org/wiki/Web_colors
    */
  final val palette
  : IndexedSeq[Color] = IndexedSeq(
    new Color(  0,   0, 205), // MediumBlue
    new Color(  0, 128,   0), // Green
    new Color(255,  69,   0), // OrangeRed
    new Color(255, 215,   0), // Gold
    new Color(255,  20, 147), // DeepPink
    new Color(189, 183, 107), // DarkKhaki
    new Color( 32, 178, 170), // LightSeaGreen
    new Color(139,  69,  19), // SaddleBrown
    new Color(255, 160, 122), // LightSalmon
    new Color(147, 112, 219), // MediumPurple

    new Color( 30, 144, 255), // DodgerBlue
    new Color( 50, 205,  50), // LimeGreen
    new Color(255, 127,  80), // Coral
    new Color(188, 143, 143), // RosyBrown
    new Color(128,   0, 128), // Purple
    new Color(112, 128, 144), // SlateGray
    new Color(  0, 128, 128), // Teal
    new Color(205, 133,  63), // Peru
    new Color(220,  20,  60), // Crimson
    new Color(238, 130, 238), // Violet

    new Color(135, 206, 235), // SkyBlue
    new Color(107, 142,  35), // OliveDrab
    new Color(255, 165,   0), // Orange
    new Color(210, 180, 140), // Tan
    new Color( 75,   0, 130) // Indigo
  )

}
