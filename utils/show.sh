#!/bin/bash
echo "Start Viewing!"
curd=$(cd `dirname $0`; pwd)
cd $curd
cd ..
tensorboard --logdir "runs"