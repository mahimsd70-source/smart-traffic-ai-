from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('traffic.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, classes=[2,3,5,7])
    vehicle_count = len(results[0].boxes)
    cv2.putText(frame, f'Vehicles: {vehicle_count}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Smart Traffic - AI Counting', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()