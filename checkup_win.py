import gradio as gr
from database import Defect, paste_to_db
from main_win import pass_data
from PIL import Image, ImageDraw

# Function to save defects to the database
def save_defects(defects, selected_labels, confirmations):
    results = []
    for defect, selected_label, confirmed in zip(defects, selected_labels, confirmations):
        if confirmed:  # Only save if the defect is confirmed
            # Create a Defect instance
            defect_instance = Defect(
                image_data=defect['image'],  # Get binary data
                bbox=str(defect['bboxes']),  # Store bounding boxes as a string
                class_id=selected_label  # Use the selected class ID from the dropdown
            )
            
            # Save to the database
            paste_to_db(defect_instance)
            results.append(f"Defect saved to database: Class ID {selected_label}")

    return results

# Function to show images with bounding boxes
def show_images_with_bboxes(defects, predictions):
    images_with_bboxes = []
    
    # Check if predictions are not None and defects are available
    if predictions is None or defects is None or len(defects) == 0:
        return None  # Return None if there are no predictions or defects

    for defect in defects:
        image = defect['image']
        bboxes = defect['bboxes']  # Assuming bboxes are stored in the defect dictionary
        
        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(image)

        for bbox in bboxes:
            draw.rectangle(bbox, outline="red", width=2)  # Draw the bounding box

        images_with_bboxes.append(image)  # Append the modified image to the list

    return images_with_bboxes  # Return the list of images with bounding boxes

with gr.Blocks() as demo:
    serial_number, defects, predictions = pass_data()  # Retrieve data from main_win
    print(serial_number, defects, predictions)
    
    # Create a placeholder for images with bounding boxes
    images_output = gr.Gallery(label="Defects with Bounding Boxes", columns=1, container=True)

    # Set up the button click event to show images with bounding boxes
    #b = gr.Button("Show Images with BBoxes")
    #b.click(show_images_with_bboxes, inputs=[defects, predictions], outputs=images_output)  # Link button to function

    # Create a dropdown for selecting labels (example)
    selected_labels = gr.Dropdown(label="Select Class ID", choices=["Class 1", "Class 2", "Class 3"], multiselect=True)

    # Create a checkbox for confirmation
    confirmations = gr.CheckboxGroup(label="Confirm Defects", choices=["Confirm"], value=["Confirm"])

    # Button to save defects
    #save_button = gr.Button("Подтвердить дефект")
    #save_button.click(save_defects, inputs=[defects, selected_labels, confirmations], outputs=images_output)  # Pass inputs as a list

iface2 = demo