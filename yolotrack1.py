import cv2
from ultralytics import YOLO, solutions



# Load YOLOv8 model
model = YOLO("yolov10n.pt")
# Initialize global variable to store cursor coordinates
line_pts = [(0, 288), (1019, 288)]
names = model.model.names

speed_obj = solutions.SpeedEstimator(reg_pts=line_pts,names=names)

# Mouse callback function to capture mouse movement
def RGB(event, x, y, flags, param):
    global cursor_point
    if event == cv2.EVENT_MOUSEMOVE:  # Detect mouse movement
        cursor_point = (x, y)  # Update cursor coordinates
        print(f"Mouse coordinates: {cursor_point}")  # Print the coordinates

# Set up the window and attach the mouse callback function
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)




# Open the video file or webcam feed
cap = cv2.VideoCapture('vid.mp4')



count = 0
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        print("Video stream ended or cannot be read.")
        break

    count += 1
    if count % 3 != 0:  # Skip some frames for speed (optional)
        continue

    # Resize the frame
    frame = cv2.resize(frame, (1020, 500))
    tracks = model.track(frame, persist=True)

    im0 = speed_obj.estimate_speed(frame, tracks)
   

    # Display the frame with YOLOv8 results, speed estimations, and cursor coordinates
    cv2.imshow("RGB", frame)

    # Use cv2.waitKey(1) to process frames continuously
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit the loop
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
