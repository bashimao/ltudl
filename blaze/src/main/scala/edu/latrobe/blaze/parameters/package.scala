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

import edu.latrobe._
import scala.language.implicitConversions

package object parameters {

  object Implicits {

    final implicit def realToParameter(r: Real)
    : Unit = ConstantValueBuilder(r)

  }

  final implicit class ParameterFunctions(p: ParameterBuilder) {

    def mirror(value: Real)
    : MirrorBuilder = MirrorBuilder(p, value)

    def clip(range: RealRange)
    : ClipBuilder = ClipBuilder(p, range)

    def clip(min: Real, max: Real)
    : ClipBuilder = ClipBuilder(p, min, max)

  }

}
