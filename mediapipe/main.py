import cv2
import mediapipe as mp

def extract_face_features(image_path):
    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2)
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results_mesh = face_mesh.process(image_rgb)

    if results_mesh.multi_face_landmarks:
        for face_landmarks in results_mesh.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

            for i, landmark in enumerate(face_landmarks.landmark):
                x_pixel, y_pixel = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                print(f"Landmark {i}: ({x_pixel}, {y_pixel})")

        cv2.imshow("Face Features", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    extract_face_features('images/image1.jpg')
