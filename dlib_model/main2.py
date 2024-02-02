import os
import cv2

# Load the face detector (Haarcascades) from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to the folder containing your images
folder_path = "images/train/madonna"

# Iterate through each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        # Load the image
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # Convert the image to grayscale for Haarcascades
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use Haarcascades to detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Iterate through the detected faces
        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the image with rectangles around detected faces
        cv2.imshow("Detected Faces", image)
        cv2.waitKey(0)

# Close the OpenCV window after processing all images
cv2.destroyAllWindows()
