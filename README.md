# FaceTrackerSampleApp


### How to clone and build this project
```bash
$ git clone git@github.com:chicn/FaceTrackerSampleApp.git

# get dlib
$ git submodule init
$ git submodule update

# Install cocoapod & get opencv
$ bundle install --path=vendor/bundle
$ bundle exec pod install
```

Download dlib model from [here](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2) and save and add it somewhere in this project.
```
$ mkdir ./vendor/dlib-model \
&&  curl -OL https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2 \
&& mv shape_predictor_68_face_landmarks.dat.bz2 ./vendor/dlib-model/ \
&& bzip2 -vd ./vendor/dlib-model/shape_predictor_68_face_landmarks.dat.bz2
```
