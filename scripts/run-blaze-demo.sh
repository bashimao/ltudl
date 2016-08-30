#!/usr/bin/env bash

if [ -z $1 ]; then
  echo You must supply the name of main class!
  echo Example: ./run-blaze-demo.sh edu.latrobe.demos.mnist.SimpleMLP
  exit -1
fi


CLASS_PATH=`find ../out/*.jar`
CLASS_PATH=`echo $CLASS_PATH | tr " " ":"`

# echo $CLASS_PATH

java -classpath $CLASS_PATH $1

