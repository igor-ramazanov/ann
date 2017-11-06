# ann
Simple and manually written artificial neural network for recognition of emotions of drawn faces.

Written in Python with NumPy library.

Divides training images to 2 parts:
1. Training set - 70%
2. Testing set - 30%

Number of layers: 2

Doesn't use back propagation.

Stopping criteria for learning: precision (percentage of properly predicted) should be greater than 68%

## Data
Detects 4 emotion of drawn faces:
1. Happy
2. Sad
3. Mischievous 
4. Mad

![Faces](https://1.downloader.disk.yandex.ru/disk/670e20928eda62ada28464d7cef0bd7fc88160dac9526010369d3329efc7e51a/5a0093bb/oo4-nujJDR2caQgt6q-lZN25F9OZShIujfIcdw4fJzhj0g40TofFQD4hQ6d3PrCvK1rrr7R9Vrg2g0gyKsNfhA%3D%3D?uid=0&filename=faces.png&disposition=inline&hash=&limit=0&content_type=image%2Fpng&fsize=20422&hid=488b2a07aefd49ed93c98a5d316dc6d7&media_type=image&tknv=v2&etag=e129d6907af6b9357e8637c88afdbbf4)

Input image is 20x20 pixels with 32 degrees of grey level from white to black.

Image format example [training.txt](training.txt):
```
Image1
0 3 3 3 3 0 1 0 31 31 25 0 0 2 12 3 0 0 3 4
4 1 9 0 0 0 0 0 0 31 31 31 4 2 0 0 0 0 0 2
0 7 0 0 0 6 1 2 4 0 31 31 29 1 0 0 0 0 6 0
0 8 31 5 0 0 0 0 7 8 0 31 31 31 6 0 0 0 0 0
0 29 31 3 7 0 6 9 6 3 3 2 31 31 31 18 0 10 0 9
3 9 31 26 5 0 4 1 0 2 9 0 5 31 21 4 0 1 2 0
2 0 31 31 4 0 3 0 0 6 3 0 1 0 0 3 0 0 6 1
0 13 31 31 8 2 1 0 0 3 8 10 1 4 0 1 0 5 0 0
0 0 31 31 1 3 10 2 4 0 5 0 6 0 0 2 0 4 0 4
8 3 21 31 0 0 5 10 0 0 7 0 0 3 2 2 0 1 0 6
1 6 0 31 21 8 6 0 6 10 3 4 0 5 0 8 8 14 9 7
0 5 0 24 8 2 0 0 0 0 0 9 2 9 9 9 6 0 10 19
0 0 8 1 0 0 0 0 4 5 14 4 14 22 0 6 0 0 6 1
0 2 0 0 0 0 7 5 11 7 7 0 9 3 3 3 0 0 9 5
1 0 0 0 4 0 1 7 13 5 2 0 13 2 6 2 0 9 0 0
3 0 11 0 8 0 8 6 2 0 4 0 3 0 1 0 8 2 0 0
7 0 0 4 0 3 4 3 3 0 3 0 8 5 5 0 0 0 1 0
3 4 0 0 0 13 9 8 0 7 2 0 7 2 0 0 2 4 7 10
0 1 0 4 0 12 4 0 0 0 0 0 0 6 0 8 0 0 8 0
0 0 0 9 3 2 23 0 0 0 1 0 1 6 0 6 3 0 0 6
```

Emotions [training-answers.txt](training-answers.txt):
```
Image1 2
Image2 1
Image3 4
Image4 4
...
```

## Dependencies
Python2.7 or Python3.6, NumPy

## Usage
```
./faces.py \
<path to file with training images> \
<path to file with answers for training images> \
<path to file with testing images>
```
