# Face landmarks extraction

In this we use media pipe and opencv to detect face landmarks and tries to use those data for authenticating a user. This is done as part of the project <a href='https://github.com/aswanthabam/VoteChain'>VoteChain</a>.

## How does it worK?

- First of all we find the face landmarks from an image (here we use live camera visual)
- next we will normalize this landmark points with relation to a perticular point 
  - many docs and research says using the nose coordinates as the centroid would be better
- want to find the most relevent points in the landmarks and classify them
- use this normalized points to get the feature vector
- we plan to generate a hash from this coordinates and this hash will be used to verify the person
- this hash will be send to the verifier
  - it will verify and find how much does the registered face and the given face match,(in %)
  - according to the threashold value the system will decide if the two faces are match


## Todo

- [ ] Normalize Landmark Coordinates:

      Normalize the coordinates of facial landmarks to be invariant to scale, translation, and rotation. This helps in making the system more robust to variations in pose and facial expression.
- [ ] Create a Feature Vector:

      Concatenate or calculate the distances between certain landmark points to create a feature vector for each face.

## References

- [Google Developers Guide](https://developers.google.com/mediapipe/solutions/vision/face_landmarker/python)
- [face landmarks recognition - italojs](https://github.com/italojs/facial-landmarks-recognition)
- [Medium Blog on face landmark and KNN](https://medium.com/@ragilprasetyo310/simple-face-recognition-with-facial-landmark-k-nearest-neighbors-ad5ae733adba#:~:text=Facial%20Landmark%20refer%20to%20specific,mouth%2C%20and%20other%20facial%20structures.)