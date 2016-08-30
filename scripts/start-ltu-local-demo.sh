#!/usr/bin/env bash

if [ -z $1 ]; then
  echo "No class name given."
  echo "Example: start-ltu-local-demo.sh edu.latrobe.demos.imagenet.TrainBlazeModel"
  exit -1
fi

./start-app.sh ~/debug LOCAL demos-0.3-SNAPSHOT.jar $1
