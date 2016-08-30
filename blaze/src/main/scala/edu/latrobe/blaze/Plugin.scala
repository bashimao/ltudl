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

package edu.latrobe.blaze

import java.util.concurrent.Semaphore

import edu.latrobe._
import org.json4s.JsonAST._

import scala.collection._

/**
  * Name must be unique!
  */
abstract class Plugin
  extends Equatable
    with Serializable {

  def name: String

  def platforms
  : Set[IndependentPlatform]

  final def load()
  : Unit = {
    locking(Plugin._pluginsGuard)({
      // Avoid double registration.
      if (Plugin.isRegistered(name)) {
        logger.error(s"$name has already been registered!")
        return
      }

      // Register platforms.
      platforms.foreach(IndependentPlatform.register)

      // Register variants and similar stuff.
      doLoad()

      // Add to plugin list.
      Plugin._plugins = this :: Plugin._plugins
    })
  }

  protected def doLoad()
  : Unit

  final def unload()
  : Unit = {
    locking(Plugin._pluginsGuard)({
      // Avoid double un-registration.
      if (!Plugin.isRegistered(name)) {
        logger.error(s"$name was not registered!")
        return
      }

      // Unregister variants and similar stuff.
      doUnload()

      // Unregister platforms.
      platforms.foreach(IndependentPlatform.unregister)

      // Remove from plugin list.
      Plugin._plugins = Plugin._plugins.filter(_.name == name)
    })
  }

  protected def doUnload()
  : Unit

  def collectRuntimeStatus()
  : JValue

}

object Plugin {

  private val _pluginsGuard
  : Semaphore = new Semaphore(1)

  final private var _plugins
  : List[Plugin] = Nil

  final def plugins
  : List[Plugin] = _plugins

  final def getPlugin(name: String)
  : Option[Plugin] = _plugins.find(_.name == name)

  final def isRegistered(name: String)
  : Boolean = _plugins.exists(_.name == name)

  final def isRegistered(plugin: Plugin)
  : Boolean = isRegistered(plugin.name)

}
