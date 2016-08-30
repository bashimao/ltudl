/*
 * La Trobe University - Distributed Deep Learning System
 * Copyright 2014 Matthias Langer (t3l@threelights.de)
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
 */

package edu.latrobe.demos.util.imagenet

import au.com.bytecode.opencsv._
import edu.latrobe._
import edu.latrobe.blaze._
import edu.latrobe.sizes._
import edu.latrobe.io._
import java.io._
import org.json4s._
import org.json4s.jackson._
import org.w3c.dom._
import scala.util.matching._

object ImageNetUtils {

  final def loadSynsets(stream: InputStream)
  : Array[Synset] = {
    val synsets = Array.newBuilder[Synset]
    readSynsets(stream, synsets += _)
    synsets.result()
  }

  final def readSynsets(stream: InputStream, callback: Synset => Unit)
  : Unit = {

    def parseEntry(iter: Iterator[String])
    : Synset = {
      val imageNetID = iter.next().toInt

      val wordNetID = iter.next()

      val words = {
        val noWords = iter.next().toInt
        ArrayEx.fill(
          noWords
        )(iter.next())
      }

      val gloss = {
        val noGloss = iter.next().toInt
        ArrayEx.fill(
          noGloss
        )(iter.next())
      }

      val children = {
        val noChildren = iter.next().toInt
        ArrayEx.fill(
          noChildren
        )(iter.next().toInt)
      }

      val wordNetHeight = iter.next().toInt

      val noTrainImages = iter.next().toInt

      Synset(
        imageNetID, wordNetID,
        words, gloss, children,
        wordNetHeight, noTrainImages
      )
    }

    // Load data.
    val reader = new CSVReader(
      new BufferedReader(new InputStreamReader(stream)),
      ',',
      '"',
      '\\',
      0
    )
    var values = reader.readNext()
    while (values != null) {
      // Skip empty lines.
      if (values.length > 0 && values.head.nonEmpty) {
        callback(parseEntry(values.iterator))
      }
      values = reader.readNext()
    }
  }

  final def loadSynsets(file: FileHandle)
  : Array[Synset] = {
    val builder = Array.newBuilder[Synset]
    using(
      file.openStream()
    )(readSynsets(_, builder += _))
    builder.result()
  }

  final private val xmlFilePattern
  : Regex = """.*\.xml$""".r

  final def readAnnotations(stream: InputStream, callback: Annotation => Unit)
  : Unit = {
    val doc    = XML.parse(stream)
    val nodes  = doc.getElementsByTagName("annotation")
    val length = nodes.getLength
    var i      = 0
    while (i < length) {
      val annotation = Annotation.derive(nodes.item(i).asInstanceOf[Element])
      callback(annotation)
      i += 1
    }
  }

  final def loadAnnotations(basePath: FileHandle)
  : Array[Annotation] = {
    val annotations = Array.newBuilder[Annotation]
    basePath.traverse(
      (depth, handle) => true,
      (depth, handle) => {
      if (handle.matches(xmlFilePattern)) {
        using(
          handle.openStream()
        )(readAnnotations(_, annotations += _))
      }
    })
    annotations.result()
  }

  final def loadTrainingSet(basePath:  FileHandle,
                            dirName:   String,
                            noClasses: Int,
                            cache:     Boolean)
  : Array[Batch] = {
    // Synsets.
    val synsets = {
      val tmp     = loadSynsets(basePath ++ "meta_clsloc.csv")
      val builder = Map.newBuilder[String, Synset]
      ArrayEx.foreach(
        tmp
      )(s => builder += Tuple2(s.wordNetID, s))
      builder.result()
    }
    logger.info(s"${synsets.size} synsets found!")

    // Annotations.
    val annotations = {
      val tmp     = loadAnnotations(basePath ++ "bbox_train_aggregated")
      val builder = Map.newBuilder[String, Annotation]
      ArrayEx.foreach(
        tmp
      )(a => builder += Tuple2(a.filename, a))
      builder.result()
    }
    logger.info(s"${annotations.size} annotations found!")

    // Image files.
    val files = (basePath ++ dirName).listFiles(
      (depth, handle) => handle.matches(".*\\.JPEG$".r)
    )
    logger.info(s"${files.length} images found!")

    // Create batches.
    val batchesWithNulls = ArrayEx.mapParallel(files)(fileIn => {
      val file     = if (cache) CachedFileHandle(fileIn) else fileIn
      val fileName = file.fileNameWithoutExtension
      val wnID     = fileName.split("_")(0)
      val synset   = synsets(wnID)
      val classNo  = synset.imageNetID - 1

      if (classNo < noClasses) {
        Batch(
          FileTensor.derive(Size2.zero, file),
          SparseRealMatrixTensor(MatrixEx.labelsToSparse(noClasses, classNo)),
          Array((fileName, annotations.get(fileName), synset))
        )
      }
      else {
        null
      }
    })
    val batches = ArrayEx.filter(
      batchesWithNulls
    )(_ != null)
    logger.info(s"${batches.length} batches created!")

    batches
  }

  final def loadValidationSet(basePath:  FileHandle,
                              dirName:   String,
                              noClasses: Int,
                              cache:    Boolean)
  : Array[Batch] = {
    // Synsets.
    val synsets = {
      val tmp     = loadSynsets(basePath ++ "meta_clsloc.csv")
      val builder = Map.newBuilder[String, Synset]
      ArrayEx.foreach(
        tmp
      )(s => builder += Tuple2(s.wordNetID, s))
      builder.result()
    }
    logger.info(s"${synsets.size} synsets found!")

    // Annotations.
    val annotations = {
      val tmp     = loadAnnotations(basePath ++ "bbox_val_aggregated.xml")
      val builder = Map.newBuilder[String, Annotation]
      ArrayEx.foreach(
        tmp
      )(a => builder += Tuple2(a.filename, a))
      builder.result()
    }
    logger.info(s"${annotations.size} annotations found!")

    // Image files.
    val files = (basePath ++ dirName).listFiles(
      (depth, handle) => handle.matches(".*\\.JPEG$".r)
    )
    logger.info(s"${files.length} images found!")

    // Create batches.
    val batchesWithNulls = ArrayEx.mapParallel(files)(fileIn => {
      val file       = if (cache) CachedFileHandle(fileIn) else fileIn
      val fileName   = file.fileNameWithoutExtension
      val annotation = annotations(fileName)

      var wnID: String = null
      for (body <- annotation.bodies) {
        wnID match {
          case body.name =>
            // do nothing
          case null =>
            wnID = body.name
          case _ =>
            throw new MatchError(wnID)
        }
      }

      val synset  = synsets(wnID)
      val classNo = synset.imageNetID - 1

      if (classNo < noClasses) {
        Batch(
          FileTensor.derive(Size2.zero, file),
          SparseRealMatrixTensor(MatrixEx.labelsToSparse(noClasses, classNo)),
          Array((fileName, annotation, synset))
        )
      }
      else {
        null
      }
    })
    val batches = ArrayEx.filter(
      batchesWithNulls
    )(_ != null)
    logger.info(s"${batches.length} batches created!")

    batches
  }

  final def loadTestSet(basePath: FileHandle,
                        dirName:  String,
                        cache:    Boolean)
  : Array[Batch] = {
    // Image files.
    val files = (basePath ++ dirName).listFiles(
      (depth, handle) => handle.matches(".*\\.JPEG$".r)
    )
    logger.info(s"${files.length} images found!")

    // Create batches.
    val batches = ArrayEx.mapParallel(
      files
    )(fileIn => {
      val file = if (cache) CachedFileHandle(fileIn) else fileIn
      Batch(FileTensor.derive(Size2.zero, file))
    })
    logger.info(s"${batches.length} batches created!")

    batches
  }

  final def loadMeanAndVariance(file: FileHandle)
  : Array[MeanAndVariance] = {
    using(
      file.openStream()
    )(loadMeanAndVariance)
  }

  final def loadMeanAndVariance(stream: InputStream)
  : Array[MeanAndVariance] = {
    val json = {
      val tmp = StreamEx.readText(stream)
      JsonMethods.parse(StringInput(tmp))
    }
    Array(
      MeanAndVariance.derive(
        json.findField(_._1 == "b").get._2.asInstanceOf[JObject]
      ),
      MeanAndVariance.derive(
        json.findField(_._1 == "g").get._2.asInstanceOf[JObject]
      ),
      MeanAndVariance.derive(
        json.findField(_._1 == "r").get._2.asInstanceOf[JObject]
      )
    )
  }

}
