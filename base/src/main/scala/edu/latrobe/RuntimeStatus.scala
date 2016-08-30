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

package edu.latrobe

import edu.latrobe.io._
import edu.latrobe.io.image._
import edu.latrobe.io.showoff._
import org.json4s.JsonAST._
import scala.collection.parallel._
import scala.util._
import scala.language.postfixOps
import sys.process._

object RuntimeStatus {

  final def collect()
  : JObject = {
    logger.trace("Collecting runtime status for edu.latrobe")

    val fields = List.newBuilder[JField]

    fields += Json.field("logger.isTraceEnabled", logger.isTraceEnabled)
    fields += Json.field("logger.isDebugEnabled", logger.isDebugEnabled)
    fields += Json.field("logger.isInfoEnabled", logger.isInfoEnabled)
    fields += Json.field("logger.isWarnEnabled", logger.isWarnEnabled)
    fields += Json.field("logger.isErrorEnabled", logger.isErrorEnabled)

    // --- System --------------------------------------------------------------
    fields += Json.field("javaVersion", Properties.javaVersion)
    fields += Json.field("javaVMVersion", Properties.javaVmVersion)
    fields += Json.field("scalaVersion", Properties.versionString)
    fields += Json.field("osName", Properties.osName)
    fields += Json.field("osArchitecture", Properties.propOrElse("os.arch", ""))
    fields += Json.field("osVersion", Properties.propOrElse("os.version", ""))

    fields += Json.field("noCPUCores", Runtime.getRuntime.availableProcessors())
    fields += Json.field("forkJoinParallelism", ForkJoinTasks.defaultForkJoinPool.getParallelism)
    fields += Json.field("memorySize", Runtime.getRuntime.maxMemory())
    fields += Json.field("freeMemorySize", Runtime.getRuntime.freeMemory())

    fields += Json.field("PATH", Environment.get("PATH", "undefined"))
    fields += Json.field("CLASSPATH", Environment.get("CLASSPATH", "undefined"))
    fields += Json.field("java.library.path", Properties.propOrElse("java.library.path", ""))

    // --- LTU -----------------------------------------------------------------
    fields += Json.field("Real.size", Real.size)

    fields += Json.field("LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE", LTU_REAL_TENSOR_DEFAULT_CHUNK_SIZE)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ABS", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ABS)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ADD", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_ADD)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_DIVIDE", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_DIVIDE)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_DOT", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_DOT)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_FILL", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_FILL)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_L1_NORM", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_L1_NORM)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_L2_NORM", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_L2_NORM)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_LERP", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_LERP)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MAX", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MAX)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MIN", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MIN)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MULTIPLY", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_MULTIPLY)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SET", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SET)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SIGN", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SIGN)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SQR", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SQR)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SQRT", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SQRT)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SUM", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_SUM)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_TABULATE", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_TABULATE)
    fields += Json.field("LTU_REAL_TENSOR_CHUNK_SIZE_FOR_TRANSFORM", LTU_REAL_TENSOR_CHUNK_SIZE_FOR_TRANSFORM)

    fields += Json.field("LTU_REDUNDANT_CALL_TO_CLOSE_WARNING", LTU_REDUNDANT_CALL_TO_CLOSE_WARNING)

    // --- LTU.IO --------------------------------------------------------------
    fields += Json.field("Host.name", Host.name)
    fields += Json.field("Host.address", Host.address)
    fields += Json.field("LTU_IO_FILE_HANDLE_RETHROW_EXCEPTIONS_DURING_TRAVERSE", LTU_IO_FILE_HANDLE_RETHROW_EXCEPTIONS_DURING_TRAVERSE)

    // --- LTU.IO.IMAGE --------------------------------------------------------
    fields += Json.field("LTU_IO_IMAGE_DEFAULT_IMPLEMENTATION", LTU_IO_IMAGE_DEFAULT_IMPLEMENTATION)

    // --- LTU.IO.SHOWOFF ------------------------------------------------------
    fields += Json.field("LTU_IO_SHOWOFF_HOST_ADDRESS", LTU_IO_SHOWOFF_HOST_ADDRESS)
    fields += Json.field("LTU_IO_SHOWOFF_HOST_PORT", LTU_IO_SHOWOFF_HOST_PORT)

    // --- NETLIB --------------------------------------------------------------
    fields += Json.field("LIBBLAS.alternatives", "update-alternatives --display libblas.so" !!)
    fields += Json.field("LIBBLAS3.alternatives", "update-alternatives --display libblas.so.3" !!)
    fields += Json.field("LIBLAPACK.alternatives", "update-alternatives --display libblas.so" !!)
    fields += Json.field("LIBLAPACK3.alternatives", "update-alternatives --display liblapack.so.3" !!)

    fields += Json.field("blasClassName", _BLAS.blas.getClass.getName)
    fields += Json.field("lapackClassName", _LAPACK.lapack.getClass.getName)

    fields += Json.field("OMP_NUM_THREADS", Environment.get("OMP_NUM_THREADS", "undefined"))
    fields += Json.field("OMP_DYNAMIC", Environment.get("OMP_NUM_THREADS", "undefined"))
    fields += Json.field("MKL_NUM_THREADS", Environment.get("MKL_NUM_THREADS", "undefined"))
    fields += Json.field("MKL_DYNAMIC", Environment.get("MKL_DYNAMIC", "undefined"))
    fields += Json.field("MKL_CBWR", Environment.get("MKL_CBWR", "undefined"))

    JObject(fields.result())
  }

}
