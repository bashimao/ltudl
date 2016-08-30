package edu.latrobe.native

import edu.latrobe._
import org.bytedeco.javacpp._

/**
  * Simple wrapper to make javacpp pointer support our closing method.
  */
abstract class AutoClosingPointer
  extends AutoClosing {

  def ptr
  : Pointer

}

abstract class AutoClosingPointerCompanion {

  def allocate(length: Long)
  : AutoClosingPointer

}

abstract class AutoClosingPointerEx[TPtr <: Pointer]
  extends AutoClosingPointer {

  /**
    * Should override with val!
    */
  protected def _ptr
  : TPtr

  final override def ptr
  : TPtr = {
    if (closed) {
      throw new NullPointerException
    }
    else {
      _ptr
    }
  }

}

abstract class AutoClosingPointerExCompanion[T <: AutoClosingPointerEx[TPtr], TPtr <: Pointer, TValue]
  extends AutoClosingPointerCompanion {

  override def allocate(length: Long)
  : T

  def apply(value: TValue)
  : T

  def derive(array: Array[TValue])
  : T

  /**
    * aka. NULL ptr
    */
  def NULL
  : T

}
