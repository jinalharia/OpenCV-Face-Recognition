##Description

This is a set of scripts to implement face recognition in pictures / videos.  It uses a pre-trained Caffe deep learning model to detect faces. This model detects and localizes faces in an image.  It also uses a Torch deep learning model which produces the 128-D facial embeddings.  Finally it trains a Linear SVM model. We’ll detect faces, extract embeddings, and fit our SVM model to the embeddings data.

##Setup

Create virtual environment and install requirements

```python
python -m venv facerecog
source facerecog/activate
pip install -r requirements.txt
```

##Usage
1 Put known images in dataset folder in subfolders with correct name (ensure these photos have only the named person in them)
2 Put images to test in images folder
3 Create output folder to save pickled models
4 First extract embeddings by running:
```shell
python extract_embeddings.py --dataset dataset \
	--embeddings output/embeddings.pickle \
	--detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7
```

5 Train the model
```shell
python train_model.py --embeddings output/embeddings.pickle \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle
```

6 Run script on actual images
```shell
python recognize.py --detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle \
	--image images/obama.jpg
```

7 To run on a live video stream use
```shell
python recognize_video.py --detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle
```

8 To run on folder of images use
```shell
python recognize.py --detector face_detection_model \
	--embedding-model openface_nn4.small2.v1.t7 \
	--recognizer output/recognizer.pickle \
	--le output/le.pickle \
	--images images
```