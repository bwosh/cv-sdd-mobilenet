import cv2

cap = cv2.VideoCapture(0)
print(cv2.__version__)

model = cv2.dnn.readNetFromTensorflow('./models/frozen_inference_graph.pb', './models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

def id_class_name(class_id, classes):
    for key,value in classes.items():
        if class_id == key:
            return value

while True:
    ret, frame = cap.read()
    rows = frame.shape[0]
    cols = frame.shape[1]

    model.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True))
    output = model.forward()

    for detection in output[0, 0, :, :]:
        class_id = detection[1]
        confidence = detection[2]

        if confidence > 0.5:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)


    cv2.imshow("Window", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
