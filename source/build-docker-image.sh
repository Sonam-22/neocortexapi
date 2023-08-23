#/bin/sh
mkdir -p temp
cp ../license.txt temp/license.txt
docker build . -f MyCloudProjectSample/MyCloudProject/Dockerfile -t sdrclassifier.azurecr.io/cloudcomputing/sdrclassifier:v1
rm -rf temp