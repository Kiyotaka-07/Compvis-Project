import cv2
import os
import numpy as np
import pickle


def train_model():

    dataset_path = "dataset"

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_dict = {}

    label_id = 0

    for user in os.listdir(dataset_path):

        user_path = os.path.join(dataset_path, user)

        if not os.path.isdir(user_path):
            continue

        label_dict[label_id] = user

        for img_name in os.listdir(user_path):

            img_path = os.path.join(user_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            faces.append(img)
            labels.append(label_id)

        label_id += 1

    recognizer.train(faces, np.array(labels))

    recognizer.save("face_model.yml")

    # simpan label dictionary
    with open("labels.pkl", "wb") as f:
        pickle.dump(label_dict, f)

    print("Training selesai")


if __name__ == "__main__":
    train_model()