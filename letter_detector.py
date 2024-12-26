import cv2
import numpy as np

def detect_letters(image):
    """
    Detects letter-like regions in the input image.

    Args:
        image: grayscale NumPy image.

    Returns:
        boxes: List of bounding boxes in (x0, y0, x1, y1) format.
    """
    
    # Convert to binary image for contour detection
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract and filter valid bounding boxes
    min_size = 20
    boxes = [
        (x, y, x + w, y + h)
        for contour in contours
        for x, y, w, h in [cv2.boundingRect(contour)]
        if w > min_size and h > min_size
    ]
    
    return sorted(boxes, key=lambda box: (box[0], box[1]))

def merge_nearby_boxes(boxes, max_distance=30):
    """
    Merge bounding boxes that are close to each other vertically.

    Args:
        boxes (list of tuples): A list of bounding boxes, where each box is represented 
                                as a tuple (x1, y1, x2, y2). 
                                - (x1, y1): Top-left corner of the box.
                                - (x2, y2): Bottom-right corner of the box.
        max_distance (int): The maximum vertical distance between two boxes for them 
                            to be considered "close" and eligible for merging. Default is 20.

    Returns:
        list of tuples: A list of merged bounding boxes.

    Explanation:
        Horizontal Overlap:
        -------------------
        Two boxes are considered to overlap horizontally if their projections on the 
        x-axis intersect. Mathematically, horizontal overlap exists if:
        
            min(x2_box1, x2_box2) > max(x1_box1, x1_box2)
            min of the right boundaries of the boxes is greater than the max of the left boundaries.

        Where:
            - x1_box1 and x2_box1 are the left and right boundaries of the first box.
            - x1_box2 and x2_box2 are the left and right boundaries of the second box.

        The overlap width is calculated as:
        
            overlap_width = min(x2_box1, x2_box2) - max(x1_box1, x1_box2)

        If overlap_width > 0, the boxes overlap horizontally.

        Visual Representation of Horizontal Overlap:
        --------------------------------------------
        Case 1: Overlapping boxes
            Box 1: |---10---[Box1]---30---|
            Box 2:          |---20---[Box2]---40---|

            Overlapping region is between 20 and 30 (width = 10).

        Case 2: Non-overlapping boxes
            Box 1: |---10---[Box1]---30---|
            Box 2:                             |---40---[Box2]---60---|

            No overlap as there’s no intersection on the x-axis.

        Merging Conditions:
        -------------------
        Two boxes are merged if:
            1. They overlap horizontally (as defined above).
            2. The vertical distance between the bottom of one box and the top of the 
               other is less than or equal to max_distance.

        The merged box is the smallest bounding rectangle that encompasses both boxes.
    """
    if not boxes:
        return []

    merged = []
    current_box = list(boxes[0])
    
    for box in boxes[1:]:
        # Check if boxes should be merged based on proximity
        x_overlap = (min(current_box[2], box[2]) - max(current_box[0], box[0])) > 0
        y_distance = abs(box[1] - current_box[3])
        
        if x_overlap and y_distance <= max_distance:
            # Expand current box to encompass both boxes
            current_box = [
                min(current_box[0], box[0]),  # x0
                min(current_box[1], box[1]),  # y0
                max(current_box[2], box[2]),  # x1
                max(current_box[3], box[3])   # y1
            ]
        else:
            merged.append(tuple(current_box))
            current_box = list(box)
    
    merged.append(tuple(current_box))
    return merged


def pad_and_center_image(img):
    """Pad the image to make it square and center it with white (255) around."""

    h, w = img.shape
    
    # Make image square by padding the shorter dimension
    if h > w:
        pad_left = (h - w) // 2
        pad_right = h - w - pad_left
        padded = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, 
                                  cv2.BORDER_CONSTANT, value=255)
    elif w > h:
        pad_top = (w - h) // 2
        pad_bottom = w - h - pad_top
        padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, 
                                  cv2.BORDER_CONSTANT, value=255)
    else:
        padded = img
    
    # Add uniform padding around the square image
    padding = 15
    return cv2.copyMakeBorder(padded, padding, padding, padding, padding,
                            cv2.BORDER_CONSTANT, value=255)