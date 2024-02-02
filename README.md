# Face Verification using python

in this repo we use dlib and resnet model for face verification.

- first we extract the face landmarks and from that we extract the ROI
- then we use the resnet model for extracting face embeddings, which is a 128 length array
- to verify two faces, the embeddings is compared using cosine similarity. if the similarity is greater than 0.9 the faces are match else its not

> the code for testing the algorithm is in `main.ipynb`

> Your suggestions and queries matters :)
