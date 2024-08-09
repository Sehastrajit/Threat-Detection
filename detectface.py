import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Dictionary to map class indices to sign language alphabet
class_to_alphabet = {
    0: 'Threat',
    1: 'Safe',
}

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Access webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Run YOLO inference only on detected faces
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h, x:x+w]

        # Run inference
        results = model(face_frame)

        # Process detections
        boxes_and_scores = []
        for detection in results:
            boxes = detection.boxes
            for box, cls, conf in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
                class_id = int(cls)
                class_name = class_to_alphabet.get(class_id, 'Unknown')
                x1, y1, x2, y2 = [int(coord) for coord in box]
                boxes_and_scores.append((class_name, conf, x1, y1, x2, y2))

        # Non-maximum suppression to remove overlapping bounding boxes
        boxes_and_scores.sort(key=lambda x: x[1], reverse=True)
        used_boxes = set()
        for class_name, conf, x1, y1, x2, y2 in boxes_and_scores:
            if (x1, y1, x2, y2) not in used_boxes:
                # Adjust coordinates relative to the original frame
                x1, y1, x2, y2 = x1 + x, y1 + y, x2 + x, y2 + y
                # Draw bounding box and label on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                used_boxes.add((x1, y1, x2, y2))

    # Display the frame with bounding boxes and labels
    cv2.imshow('Sign Language Detection', frame)

    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
