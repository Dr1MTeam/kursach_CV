import gradio as gr

from model_inference import predict_image, load_model
import numpy as np
import cv2
from PIL import Image, ImageDraw

model = load_model()

global predictions  # Declare the use of the global variable
predictions = []
defects = []   
serial_number = ""  # Добавлено поле для серийного номера

def predict_multiple_images(image_files):
    global predictions, defects  # Declare the use of the global variables
    predictions = []  # Reset the global variable for new predictions
    defects = []      # Reset the global variable for new defects
    for image_file in image_files:
        image = Image.open(image_file)  # Open the image file
        prediction, bboxes, bbox_metrics = predict_image(model, image)  # Unpack the returned tuple
        predictions.append(prediction)  # Store the prediction
        
        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            draw.rectangle(bbox, outline="red", width=2)  # Draw the bounding box
        
        # Save the modified image to defects
        defects.append({'image': image.copy(), 'bboxes': bboxes})  # Store a copy of the image and bboxes in defects

    return predictions

def show_defects():
    if not serial_number:  # Проверка на наличие серийного номера
        return "Сначала введите серийный номер."  # Сообщение об ошибке
    return [defect['image'] for defect in defects]  # Return the list of images with bounding boxes


def pass_data():
    return serial_number, defects, predictions
# Gradio interface
with gr.Blocks() as demo:

    serial_input = gr.Textbox(label="Серийный номер", placeholder="Введите серийный номер")  # Поле для серийного номера

    image_input = gr.File(label="Upload Images", type="filepath", file_count="multiple")  # Allow multiple file uploads


    predict_button = gr.Button("Загрузить картинки")
    #predictions_output = gr.Textbox(label="Predictions Output")  # Output for predictions

    result_state = gr.State([])

    predict_button.click(predict_multiple_images, inputs=image_input, outputs=result_state)

    serial_input.change(lambda x: globals().update(serial_number=x), inputs=serial_input)

# Launch the Gradio interface
iface = demo