import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import onnxruntime as ort
import cv2
from letter_detector import detect_letters, merge_nearby_boxes, pad_and_center_image

class LetterRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Letter Recognition")
        
        self.current_x = None
        self.current_y = None
        self.drawing = False
        
        # Set canvas dimensions and drawing parameters
        self.canvas_width = 800
        self.canvas_height = 400
        self.preview_width = 400
        self.preview_height = 200
        self.pen_size = 20
        
        # Initialize drawing surface
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image, mode="L")
        
        self.setup_ui()
    
    def setup_ui(self):
        # Configure main layout frames
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=5, pady=5)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, padx=5, pady=5)
        
        # Setup drawing canvas
        self.canvas = tk.Canvas(
            left_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg='white',
            cursor="crosshair"
        )
        self.canvas.pack(padx=5, pady=5)
        
        # Setup preview canvas
        self.preview_canvas = tk.Canvas(
            right_frame,
            width=self.preview_width,
            height=self.preview_height,
            bg='white'
        )
        self.preview_canvas.pack(padx=5, pady=5)
        
        # Configure mouse event bindings
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        # Setup control panel
        self._create_control_panel(main_frame)
        
    def _create_control_panel(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(control_frame, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Predict", command=self.predict_letter).pack(side=tk.LEFT, padx=5)
        
        self.prediction_label = ttk.Label(
            control_frame, text="Predicted Letter: ", font=('Arial', 14)
        )
        self.prediction_label.pack(side=tk.LEFT, padx=20)
        
        # Setup brush size control
        brush_frame = ttk.Frame(parent)
        brush_frame.grid(row=2, column=0, columnspan=2, pady=5)
        
        ttk.Label(brush_frame, text="Brush Size:").pack(side=tk.LEFT, padx=5)
        self.brush_scale = ttk.Scale(
            brush_frame,
            from_=5,
            to=50,
            orient=tk.HORIZONTAL,
            command=self.update_pen_size,
            length=200
        )
        self.brush_scale.set(20)
        self.brush_scale.pack(side=tk.LEFT, padx=5)

    def _get_points_on_line(self, x1, y1, x2, y2):
        # Generate smooth line points using linear interpolation
        points = []
        length = int(((x2-x1) ** 2 + (y2-y1) ** 2) ** 0.5)
        
        if length == 0:
            return [(x1, y1)]
        
        for i in range(length):
            t = i / length
            x = int(x1 * (1-t) + x2 * t)
            y = int(y1 * (1-t) + y2 * t)
            points.append((x, y))
        
        return points

    def start_drawing(self, event):
        self.drawing = True
        self.current_x = event.x
        self.current_y = event.y

    def draw_on_canvas(self, event):
        if not self.drawing:
            return
            
        x1, y1 = (self.current_x, self.current_y)
        x2, y2 = (event.x, event.y)
        
        # Draw smooth line on canvas
        self.canvas.create_line(
            x1, y1, x2, y2,
            width=self.pen_size,
            fill='black',
            capstyle=tk.ROUND,
            smooth=True,
            splinesteps=36
        )
        
        # Create smooth line effect on PIL image
        points = self._get_points_on_line(x1, y1, x2, y2)
        radius = self.pen_size // 2
        for x, y in points:
            self.draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=0
            )
        
        self.current_x = event.x
        self.current_y = event.y
            
    def stop_drawing(self, event):
        self.drawing = False
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.preview_canvas.delete("all")
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image, "L")
        self.prediction_label.config(text="")
        
    def update_pen_size(self, value):
        self.pen_size = int(float(value))
        
    def update_preview(self, preview_image):
        # Convert and display preview image
        photo = ImageTk.PhotoImage(preview_image)
        self.preview_canvas.delete("all")
        self.preview_canvas.image = photo
        self.preview_canvas.create_image(
            self.preview_width//2, self.preview_height//2,
            image=photo
        )

    def predict_letter(self):
        # Process the drawn image for letter recognition
        img_copy = np.array(self.image.copy())
        boxes = detect_letters(img_copy)
        merged_boxes = merge_nearby_boxes(boxes)

        # Prepare batch of letter images
        letter_images = []
        for box in merged_boxes:
            x1, y1, x2, y2 = box
            letter_img = img_copy[y1:y2, x1:x2]
            padded_img = pad_and_center_image(letter_img)
            resized_img = cv2.resize(padded_img, (28, 28), interpolation=cv2.INTER_NEAREST)
            normalized_img = resized_img.astype(np.float32) / 255.0
            letter_images.append(np.expand_dims(normalized_img, axis=0))

        if not letter_images:
            return

        letter_images_batch = np.stack(letter_images, axis=0)

        # Initialize ONNX model if not already loaded
        if not hasattr(self, 'onnx_session'):
            self.onnx_session = ort.InferenceSession('emnist.onnx')

        # Run prediction on batch
        input_name = self.onnx_session.get_inputs()[0].name
        result = self.onnx_session.run(None, {input_name: letter_images_batch})

        # Map predictions to characters
        index_to_char_map = {
            i: chr(ord('a') + i - 1) if i > 0 else 'N/A' 
            for i in range(27)
        }
        
        predictions = [index_to_char_map.get(np.argmax(output), '?') 
                      for output in result[0]]
        word = ''.join(predictions)

        # Visualize results
        detected_pil = Image.fromarray(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(detected_pil)
        
        for idx, (box, letter) in enumerate(zip(merged_boxes, predictions)):
            x1, y1, x2, y2 = box
            draw.rectangle(box, outline=(0, 255, 0), width=2)
            draw.text((x1, y1 - 10), letter, fill=(0, 255, 0))

        detected_pil.thumbnail((self.preview_width, self.preview_height), 
                             Image.Resampling.LANCZOS)
        self.update_preview(detected_pil)
        self.prediction_label.config(text=f"The Word: {word}")

def main():
    root = tk.Tk()
    app = LetterRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()