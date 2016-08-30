#!/usr/bin/env bash

cd ..
mvn compile
mvn package

rm -rf out
mkdir out
cp ./demos/target/dependency/*.jar out
cp ./demos/target/*.jar out
cp ./src/main/resources/log4j.properties out
cp ./src/main/resources/logback.xml out
