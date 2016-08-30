#!/usr/bin/env bash

#set -v

showHelp() {
  echo "start-app.sh working_directory [LOCAL|SPARK] jar_name main_class"
}

# Decode arguments.
if [ -z $1 ]; then
  echo "Argument Error 1"
  showHelp
  exit -1
elif [ "$2" != "LOCAL" ] && [ "$2" != "SPARK" ]; then
  echo "Argument Error 2"
  showHelp
  exit -2
elif [ -z $3 ]; then
  echo "Argument Error 3"
  showHelp
  exit -3
elif [ -z $4 ]; then
  echo "Argument Error 4"
  showHelp
  exit -4
fi

export WORKING_DIRECTORY=$1
export MODE=$2
export JAR_NAME=$3
export MAIN_CLASS=$4

cd $WORKING_DIRECTORY

export JAVA_PATH=/usr/lib/jvm/java-7-oracle-cloudera
export PATH=$JAVA_PATH/bin:$PATH
java -version

# Working Directory for logger, find * for the jar files
CLASS_PATH=`find *.jar`
CLASS_PATH=`echo $CLASS_PATH | tr " " ":"`

CONFIG_PATH=$PWD

LIBRARY_PATH=/usr/lib/jni:/cus/netlib-java

JAVA_DEBUG=-agentlib:jdwp=transport=dt_socket,server=y,address=50005,suspend=n
JAVA_PROF0=-Dcom.sun.management.jmxremote
JAVA_PROF1=-Dcom.sun.management.jmxremote.port=50006
JAVA_PROF2=-Dcom.sun.management.jmxremote.ssl=false
JAVA_PROF3=-Dcom.sun.management.jmxremote.authenticate=false
JAVA_PROF="$JAVA_PROF0 $JAVA_PROF1 $JAVA_PROF2 $JAVA_PROF3"

#export JIT_DIAGNOSTIC="-XX:+PrintCompilation -XX:+UnlockDiagnosticVMOptions -XX:+PrintInlining -XX:+PrintAssembly -XX:CompileOnly=TestPerformanceOfStaticVsDynamicCalls"
#export JIT_DIAGNOSTIC=-XX:+UseConcMarkSweepGC
GC_CONFIG="-XX:+UseG1GC -XX:G1HeapRegionSize=32M -XX:MaxGCPauseMillis=10000 -XX:ParallelGCThreads=16 -XX:ConcGCThreads=4 -XX:+PrintGCTimeStamps"

# MKL Configuration
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE
export MKL_CBWR=TRUE

# LTU Configuration

# Blaze Configuration
export LTU_IO_SHOWOFF_HOST_ADDRESS=131.172.127.188
export LTU_IO_IMAGE_DEFAULT_IMPLEMENTATION=OpenCV

# CUBlaze Configuration
export CUBLAZE_STAGING_BUFFER_SIZE=1073741824
export CUBLAZE_NO_LOGICAL_DEVICES=10
#export CUBLAZE_NO_LOGICAL_DEVICES=11

# Inferno Configuration
export INFERNO_SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS=-Djava.library.path=$LIBRARY_PATH


if [ "$MODE" == "LOCAL" ]; then

  #CLASS_PA
  # TH=.:$CLASS_PATH

  export INFERNO_SPARK_MASTER=spark://jabba:7077

  # Memory constraints
  JIT_XMS=16384m
  #JIT_XMS=4096m
  #JIT_XMS=2048m
  #JIT_XMX=196608m
  #JIT_XMX=131072m
  #JIT_XMX=65536m
  JIT_XMX=16384m
  #JIT_XMX=4096m
  #JIT_XMX=2048m

  # Run
  java -Xdebug $JAVA_DEBUG $JAVA_PROF -Djava.library.path=$LIBRARY_PATH -Xms$JIT_XMS -Xmx$JIT_XMX $JIT_DIAGNOSTIC -classpath $CLASS_PATH $MAIN_CLASS

elif [ "$MODE" == "SPARK" ]; then

  CLASS_PATH=`echo $CLASS_PATH | tr ":" ","`
  CLASS_PATH=`echo -n $CLASS_PATH | awk -v RS=, -v ORS=, '$0 != "'$JAR_NAME'"' | sed 's/,$//'`;

  # Memory constraints
  DRIVER_MEMORY=25600m
  #DRIVER_MEMORY=65536m
  #DRIVER_MEMORY=98304m
  #EXECUTOR_MEMORY=98304m
  #EXECUTOR_MEMORY=16384m
  EXECUTOR_MEMORY=25600m
  #EXECUTOR_MEMORY=65536m

  # Enable jabber
  #VERBOSE=--verbose
  VERBOSE=

  # General spark stuff
  export MASTER_URL=spark://jabba:7077
  #export MASTER_URL=local[1]
  export APP_NAME=Inferno_Spark_App

  # Inferno Configuration
  export DBG_EXECUTOR=-agentlib:jdwp=transport=dt_socket,server=y,address=50006,suspend=n

  #export INFERNO_SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS=$DBG_EXECUTOR

  # Ensure log4j finds properties file
  LOG_PROPS=$PWD/log4j.properties
  FILES=$LOG_PROPS

  # Run
  #CONF_LTU=-DLTU_IO_SHOWOFF_HOST_ADDRESS=$LTU_IO_SHOWOFF_HOST_ADDRESS
  #CONF_BLAZE=-DLTU_IO_IMAGE_DEFAULT_IMPLEMENTATIONN=$LTU_IO_IMAGE_DEFAULT_IMPLEMENTATION
  #CONF_CUBLAZE=-DCUBLAZE_STAGING_BUFFER_SIZE=$CUBLAZE_STAGING_BUFFER_SIZE
  #CONF_INFERNO=
  #CONF4=INFERNO_SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS=$INFERNO_SPARK_EXECUTOR_EXTRA_JAVA_OPTIONS

  DRIVER_CONF=$JAVA_DEBUG
  #EXECUTOR_CONF="spark.executor.extraJavaOptions=$CONF_LTU,$CONF_BLAZE,$CONF_CUBLAZE"

  export INFERNO_NO_AGENTS_PER_EXECUTOR_MAX=2
  #export INFERNO_NO_AGENTS_PER_EXECUTOR_MAX=1
  #export INFERNO_HOST_BLACKLIST="snootles"

  # --driver-class-path $CLASS_PATH
  # --jars $CLASS_PATH
  # -Djava.library.path=$LIBRARY_PATH"
  echo spark-submit $VERBOSE --files $FILES --master $MASTER_URL --name $APP_NAME --class $MAIN_CLASS --jars $CLASS_PATH --driver-class-path $CONFIG_PATH --driver-java-options "\""$DRIVER_CONF"\"" --driver-memory $DRIVER_MEMORY --conf "\""$EXECUTOR_CONF"\"" --executor-memory $EXECUTOR_MEMORY $JAR_NAME
  spark-submit $VERBOSE --files $FILES --master $MASTER_URL --name $APP_NAME --class $MAIN_CLASS --jars $CLASS_PATH --driver-library-path $LIBRARY_PATH --driver-class-path $CONFIG_PATH --driver-java-options "\""$DRIVER_CONF"\"" --driver-memory $DRIVER_MEMORY --executor-memory $EXECUTOR_MEMORY $JAR_NAME

else

  echo Invalid mode selected!

fi
