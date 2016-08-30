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

import java.awt.event.KeyEvent
import edu.latrobe._
import edu.latrobe.time._
import scala.collection.mutable

/**
  * Super trivial keyboard query helper.
  *
  * Source: http://www.darkcoding.net/software/non-blocking-console-io-is-not-possible/
  */
object LazyKeyboard
  extends AutoClosing {

  // TODO: Investigate why this does not work sometimes.
  /*
  /**
    *  Execute the specified command and return the output
    */
  private def exec(cmd: Array[String]): String = {
    val outStream = new FastByteArrayOutputStream
    val process   = Runtime.getRuntime.exec(cmd)

    val inpStream = process.getInputStream
    var c = inpStream.read()
    while (c != -1) {
      outStream.write(c)
      c = inpStream.read()
    }

    val errStream = process.getErrorStream
    c = errStream.read()
    while (c != -1) {
      outStream.write(c)
      c = errStream.read()
    }

    process.waitFor()
    new String(outStream.toByteArray)
  }

  /**
    *  Execute the stty command with the specified arguments
    *  against the current active terminal.
    */
  private def setTTY(args: String): String = {
    val cmd = s"stty $args < /dev/tty"
    exec(Array("sh", "-c", cmd))
  }

  // Backup state.
  private val ttyConfig = setTTY("-g")

  // Switch terminal to character-buffering mode.
  setTTY("-icanon min 1")

  // Disable character echo.
  setTTY("-echo")

  override protected def onDispose(disposing: Boolean): Unit = {
    try {
      setTTY(ttyConfig.trim)
    }
    catch {
      case e: Exception =>
    }
    super.onDispose(disposing)
  }
  */

  /**
    * If this key is pressed all previously recorded key presses are invalidated.
    */
  var resetKey
  : Int = '~'

  private val pressedKeys
  : mutable.Map[Int, Int] = mutable.Map.empty

  private var lastUpdate
  : Timestamp = Timestamp.zero

  def refresh(): Unit = {
    while (System.in.available() > 0) {
      val key = System.in.read()
      if (key == resetKey) {
        pressedKeys.clear()
      }
      else {
        val count = pressedKeys.getOrElse(key, 0)
        pressedKeys.update(key, count + 1)
      }
    }

    lastUpdate = Timestamp.now()
  }

  def keyPressed(key: Int): Boolean = {
    val now = Timestamp.now()
    if (TimeSpan(lastUpdate, now).seconds >= Real.one) {
      refresh()
    }
    val count = pressedKeys.getOrElse(key, 0)
    if (count > 0) {
      pressedKeys.update(key, count - 1)
      true
    }
    else {
      false
    }
  }

  // TODO: Not tested!
  def keyPressed(key: KeyEvent): Boolean = keyPressed(key.getKeyChar)

}
