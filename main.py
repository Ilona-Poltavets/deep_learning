import logging
import multiprocessing
from ultralytics import YOLO
import cv2


def get_frame():
    cap = cv2.VideoCapture(701)

    if not cap.isOpened():
        logging.info("Error: Could not open OBS Virtual Camera")
        exit()

    while (True):
        ret, frame = cap.read()
        if not ret:
            logging.info("Error: Could not read frame")
            break

        cv2.imshow('OBS Virtual Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_detection():
    model = YOLO('./runs/detect/animal_classification2/weights/best.pt')

    cap = cv2.VideoCapture(701)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        logging.info("Error: Could not open OBS Virtual Camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("Error: Could not read frame")
            break

        results = model(frame, stream=False,conf=0.5)

        annotated_frame = results[0].plot()

        cv2.imshow('YOLOv8 Detection (OBS Virtual Camera)', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting OBS Virtual Camera")
    run_detection()
    # q = multiprocessing.JoinableQueue()
    # p1 = multiprocessing.Process(target=get_frame, args=(), daemon=True)  # Removed 'q' from args
    # p1.start()
