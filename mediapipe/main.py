import cv2
import mediapipe as mp
import numpy as np

def normalize_landmarks(landmarks):
    landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
    centroid = np.mean(landmarks_array, axis=0)
    normalized_landmarks = landmarks_array - centroid
    scale_factor = np.linalg.norm(normalized_landmarks, axis=0).max()
    normalized_landmarks /= scale_factor
    normalized_landmarks_flat = normalized_landmarks.flatten()
    return normalized_landmarks_flat

def extract_and_normalize_face_features():
    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2)
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_mesh = face_mesh.process(frame_rgb)

        if results_mesh.multi_face_landmarks:
            for face_landmarks in results_mesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )

                normalized_landmarks = normalize_landmarks(face_landmarks.landmark)
                print("Normalized Landmarks:", normalized_landmarks)

            cv2.imshow("Face Features", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    extract_and_normalize_face_features()
