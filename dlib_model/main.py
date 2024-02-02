import dlib
import numpy as np
import os

# Load the pre-trained face recognition model from dlib
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("models\shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("models\dlib_face_recognition_resnet_model_v1.dat")

os.listdir('./images/train/')


def get_embeddings(image_url):
  image = dlib.load_rgb_image(image_url)
  # Detect faces in the image

  faces = detector(image)
  if len(faces) < 1:
    print("No face detected")
  elif len(faces)> 1:
    print("More than one face")
  else:
    face = faces[0]
    shape = sp(image, face)
    face_descriptor = facerec.compute_face_descriptor(image, shape)
    face_embedding = np.array(face_descriptor)
    return face_embedding

embedding1 = get_embeddings("images/vijay/1.jpg")
embedding2 = get_embeddings("images/vijay/2.jpg")
print(embedding1,embedding2)

import numpy as np
from scipy.spatial.distance import euclidean

similarity = 1 - euclidean(embedding1, embedding2)
print(similarity)

# Set a threshold for similarity (you may need to fine-tune this based on your data)
threshold = 0.6

# Compare similarity with the threshold to determine if faces are a match
if similarity > threshold:
    print("Faces are a match!")
else:
    print("Faces are not a match.")