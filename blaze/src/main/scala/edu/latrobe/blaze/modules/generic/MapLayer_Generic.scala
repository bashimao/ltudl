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

package edu.latrobe.blaze.modules.generic

import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.blaze.modules._

/**
 * Retains platform flags to mapping transforms.
 */
trait MapLayer_Generic[TBuilder <: MapLayerBuilder[_]]
  extends MapLayer[TBuilder] {

  final override lazy val outputPlatform
  : Platform = inputHints.platform

}

trait MapLayer_Generic_Builder[TThis <: MapLayer_Generic_Builder[_]]
  extends MapLayerBuilder[TThis] {

  final override def outputPlatformFor(hints: BuildHints)
  : Platform = hints.platform

}
