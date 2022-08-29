#! /bin/bash
# sh AutoGenIRModule/profiler/scripts/profile.sh /home/onceas/wanna/ModelSplit/AutoGenIRModule/profiler/resnet_split_gen
file_name=$1
sleepTime=$2
if [ ! $file_name ] ;then
 echo "please set file_name"
 exit 0
fi

if [ ! $sleepTime ] ; then
  sleepTime=10
fi

pyfile=${file_name}.py
profile_file=${file_name}.txt

cat /dev/null > $profile_file

echo "pyfile: "$pyfile
echo "profile_file: "$profile_file
for i in $(seq 1 15)
do
  currentTime=`date +%y-%m-%d-%X-%Z`
  echo "arg="$i"-------------------------"$currentTime  >> $profile_file
  python3 $pyfile $i &
  python3 /home/onceas/wanna/ModelSplit/AutoGenIRModule/profiler/nvidia_profile.py >> $profile_file
  sleep $sleepTime
done
