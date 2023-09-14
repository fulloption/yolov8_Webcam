import cv2
from ultralytics import YOLO
import ultralytics
cap = cv2.VideoCapture(0)

# ultralytics.checks()
model = YOLO('yolov8s.pt')
# model = YOLO('runs/detect/drawing_paper/weights/best.pt')
x_line = 600
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    if success:

        # Run YOLOv8 inference on the frame
        resized_frame = cv2.resize(frame, (1024, 700), interpolation=cv2.INTER_LINEAR)
        # cv2.line(resized_frame, (0, x_line), (width, x_line), (255, 0, 0), 10)

        # Visualize the results on the frame
        results = model(resized_frame, conf=0.6, classes=0)

        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached
        break

cap.release()
cv2.destroyAllWindows()