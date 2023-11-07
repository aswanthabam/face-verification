from deepface import DeepFace

result = DeepFace.verify(img1_path = "/home/admingct/Desktop/VoteChain/face-verification/deepface/images/image1.jpg", img2_path = "/home/admingct/Desktop/VoteChain/face-verification/deepface/images/image2.jpg")

print(result)