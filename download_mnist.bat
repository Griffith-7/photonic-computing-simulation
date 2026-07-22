@echo off
REM Download MNIST dataset for photonic computing simulation
REM Run this script from the project root directory

set DEST=data\mnist
set BASE_URL=https://ossci-datasets.s3.amazonaws.com/mnist

if not exist "%DEST%" mkdir "%DEST%"

echo Downloading MNIST dataset...

curl -L -o "%DEST%\train-images-idx3-ubyte.gz" "%BASE_URL%/train-images-idx3-ubyte.gz"
curl -L -o "%DEST%\train-labels-idx1-ubyte.gz" "%BASE_URL%/train-labels-idx1-ubyte.gz"
curl -L -o "%DEST%\t10k-images-idx3-ubyte.gz" "%BASE_URL%/t10k-images-idx3-ubyte.gz"
curl -L -o "%DEST%\t10k-labels-idx1-ubyte.gz" "%BASE_URL%/t10k-labels-idx1-ubyte.gz"

echo Decompressing...

tar -xzf "%DEST%\train-images-idx3-ubyte.gz" -C "%DEST%"
tar -xzf "%DEST%\train-labels-idx1-ubyte.gz" -C "%DEST%"
tar -xzf "%DEST%\t10k-images-idx3-ubyte.gz" -C "%DEST%"
tar -xzf "%DEST%\t10k-labels-idx1-ubyte.gz" -C "%DEST%"

echo Done! MNIST data available at %DEST%
