"""

module loads the model with the .pb extension, and allows you to get predictions from the resulting model

"""

import time
import warnings
from os import path

import numpy as np
import pandas
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import label_map_util

tf.enable_eager_execution()

export_dir = "output_directory/saved_model"
PATH_TO_LABELS = 'label_map.pbtxt'
IMAGE_PATHS = "C:/Users/inet/Downloads/test"

print('Loading model...', end='')
start_time = time.time()

# Load saved model
model = tf.saved_model.load_v2(
    export_dir=export_dir
)
detect_fn = model.signatures['serving_default']

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

warnings.filterwarnings('ignore')


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


submission = pandas.read_csv("C:/Users/inet/Downloads/submission.csv")

count = 0
with open("answer/submission.csv", "w") as output:
    output.write("image_name,domain,PredString\n")
    for index, row in submission.iterrows():
        image_path = row["image_name"] + ".png"
        print('Running inference for {}... '.format(image_path), end='')
        image_path = path.join(IMAGE_PATHS, image_path)
        image_np = load_image_into_numpy_array(image_path)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detect_fn(input_tensor)
        num_detections = int((detections.pop('num_detections')))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        boxes = []
        count_box = 0
        for box in detections['detection_boxes']:
            if detections['detection_scores'][count_box] >= 0.45:
                y_min, x_min, y_max, x_max = box
                boxes.append([
                    str(int(round(x_min * image_np.shape[1]))),
                    str(int(round(y_min * image_np.shape[0]))),
                    str(int(round(x_max * image_np.shape[1]))),
                    str(int(round(y_max * image_np.shape[0]))),
                ])
            count_box += 1
        boxes_string = "no_box"
        if len(boxes) > 0:
            boxes_string = ";".join([
                " ".join(box) for box in boxes
            ])
        output.write(",".join([
            row["image_name"],
            str(row["domain"]),
            boxes_string,
        ]) + "\n")

        print('Image #{} processed'.format(str(count)))
        count += 1
