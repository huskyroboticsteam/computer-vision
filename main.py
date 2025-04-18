from transformers import OwlViTProcessor, OwlViTForObjectDetection

from PIL import Image
import cv2
import torch

model_name = "google/owlvit-base-patch32"
processor = OwlViTProcessor.from_pretrained(model_name)
detector = OwlViTForObjectDetection.from_pretrained(model_name)

print('Models loaded')

# Define classes to detect
classes = ["hammer", "sunglasses", "a solid orange mallet or hammer"]

frame_count = 0

def try_open_camera():
    for camera_index in [0, 1, 2]:
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Successfully opened camera {camera_index}")
                return cap
            cap.release()
    return None

# Initialize webcam
cap = try_open_camera()

if cap is None:
    print("Could not access any camera. Exiting.")
    exit()
else:
    print("Press 'q' to quit...")

# Main loop
while True:
    frame_count += 1
    
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break
        
    # Convert frame to RGB for the model
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # Process frame with model
    try:
        inputs = processor(text=classes, images=pil_image, return_tensors="pt")
        outputs = detector(**inputs)
        
        # Post-process results
        target_sizes = torch.Tensor([pil_image.size[::-1]])
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
        
        # Create a copy for drawing
        display_frame = frame.copy()
        
        # Add frame counter
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw bounding boxes
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        
        # Alternative drawing method - draw all boxes in one go
        detected_anything = False
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if isinstance(score, torch.Tensor):
                score_val = score.item()
            else:
                score_val = score
                
            # No confidence threshold check - use all detections from model
            detected_anything = True
            if isinstance(box, torch.Tensor):
                box_coords = box.detach().cpu().numpy()
            else:
                box_coords = box
            
            if isinstance(label, torch.Tensor):
                label_idx = label.item()
            else:
                label_idx = label
            
            # Draw using direct numpy coordinates
            x1, y1, x2, y2 = map(int, box_coords)
            
            # Make sure coordinates are valid
            h, w = display_frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            
            # Draw with brighter color based on index
            color = (0, 255, 0)  # Green for all objects
            
            # Draw the rectangle
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw filled background for text
            text = f"{classes[label_idx]}: {score_val:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_frame, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
            
            # Add white text with black outline
            cv2.putText(display_frame, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # Black outline
            cv2.putText(display_frame, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text
            
            print(f"Drew box #{i} for {classes[label_idx]}: {score_val:.2f} at {x1},{y1},{x2},{y2}")
    
        # Display frame with detection status
        status_text = "Detecting..." if not detected_anything else f"Found {len(boxes)} objects"
        cv2.putText(display_frame, status_text, (10, display_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    except Exception as e:
        print(f"Error in model inference: {e}")
        display_frame = frame.copy()
        cv2.putText(display_frame, "Inference error", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display frame
    cv2.imshow("Object Detection", display_frame)
    
    # Exit on 'q' key press
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break

# Release resources
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
