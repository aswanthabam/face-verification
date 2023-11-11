# Face landmarks extraction

In this we use media pipe and opencv to detect face landmarks and tries to use those data for authenticating a user. This is done as part of the project <a href='https://github.com/aswanthabam/VoteChain'>VoteChain</a>.

## Todo

- [ ] Normalize Landmark Coordinates:

      Normalize the coordinates of facial landmarks to be invariant to scale, translation, and rotation. This helps in making the system more robust to variations in pose and facial expression.
- [ ] Create a Feature Vector:

      Concatenate or calculate the distances between certain landmark points to create a feature vector for each face.

## References

- [Google Developers Guide](https://developers.google.com/mediapipe/solutions/vision/face_landmarker/python)
- [face landmarks recognition - italojs](https://github.com/italojs/facial-landmarks-recognition)