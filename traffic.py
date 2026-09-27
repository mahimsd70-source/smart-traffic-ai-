from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('traffic.mp4')

print("Smart Traffic AI Started... Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video finished, restarting...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = model(frame, classes=[2,3,5,7]) # car, motorbike, bus, truck

    # Draw boxes
    annotated_frame = results[0].plot()

    vehicle_count = len(results[0].boxes)

    # Signal Logic
    if vehicle_count > 8:
        signal = "GREEN - HIGH TRAFFIC"
        color = (0,255,0)
    elif vehicle_count > 4:
        signal = "YELLOW - MEDIUM"
        color = (0,255,255)
    else:
        signal = "RED - LOW TRAFFIC"
        color = (0,0,255)

    cv2.putText(annotated_frame, f'Vehicles: {vehicle_count}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(annotated_frame, signal, (20,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Smart Traffic AI - Chennai', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()