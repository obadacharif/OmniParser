from typing import Optional
import base64
import io
import json
import os

import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_file

from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

# Initialize models
yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")
# caption_model_processor = get_caption_model_processor(model_name="blip2", model_name_or_path="weights/icon_caption_blip2")

DEVICE = torch.device('cuda')

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process():
    # Get parameters from request
    box_threshold = float(request.form.get('box_threshold', 0.05))
    iou_threshold = float(request.form.get('iou_threshold', 0.1))
    use_paddleocr = request.form.get('use_paddleocr', 'true').lower() == 'true'
    imgsz = int(request.form.get('imgsz', 640))
    
    # Get image from request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image_input = Image.open(image_file)
    
    # Process image
    box_overlay_ratio = image_input.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_input, 
        display_img=False, 
        output_bb_format='xyxy', 
        goal_filtering=None, 
        easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
        use_paddleocr=use_paddleocr
    )
    
    text, ocr_bbox = ocr_bbox_rslt
    
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_input, 
        yolo_model, 
        BOX_TRESHOLD=box_threshold, 
        output_coord_in_ratio=True, 
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config, 
        caption_model_processor=caption_model_processor, 
        ocr_text=text,
        iou_threshold=iou_threshold, 
        imgsz=imgsz
    )
    
    # Prepare results
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    
    # Format parsed content
    formatted_content = {f'icon_{i}': v for i, v in enumerate(parsed_content_list)}
    
    # Response format depends on the Accept header
    if request.headers.get('Accept') == 'application/json':
        # For API clients: return JSON with image as base64
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'parsed_content': formatted_content,
            'image_base64': img_base64
        })
    else:
        # For browser/curl: return the image directly
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return send_file(
            img_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name='processed_image.png'
        )

# Add a separate endpoint to get just the parsed content (JSON)
@app.route('/process_json', methods=['POST'])
def process_json():
    # Get parameters from request
    box_threshold = float(request.form.get('box_threshold', 0.05))
    iou_threshold = float(request.form.get('iou_threshold', 0.1))
    use_paddleocr = request.form.get('use_paddleocr', 'true').lower() == 'true'
    imgsz = int(request.form.get('imgsz', 640))
    
    # Get image from request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image_input = Image.open(image_file)
    
    # Process image (same as in process function)
    box_overlay_ratio = image_input.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_input, 
        display_img=False, 
        output_bb_format='xyxy', 
        goal_filtering=None, 
        easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
        use_paddleocr=use_paddleocr
    )
    
    text, ocr_bbox = ocr_bbox_rslt
    
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_input, 
        yolo_model, 
        BOX_TRESHOLD=box_threshold, 
        output_coord_in_ratio=True, 
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config, 
        caption_model_processor=caption_model_processor, 
        ocr_text=text,
        iou_threshold=iou_threshold, 
        imgsz=imgsz
    )
    
    # Format parsed content
    formatted_content = {f'icon_{i}': v for i, v in enumerate(parsed_content_list)}
    
    # Return JSON response
    return jsonify({
        'parsed_content': formatted_content
    })

@app.route('/', methods=['GET'])
def home():
    return '''
    <html>
        <head>
            <title>OmniParser API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                code { background-color: #f4f4f4; padding: 2px 5px; border-radius: 4px; }
                pre { background-color: #f4f4f4; padding: 15px; border-radius: 4px; overflow-x: auto; }
                .endpoint { margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <h1>OmniParser API</h1>
            <p>OmniParser is a screen parsing tool to convert general GUI screen to structured elements.</p>
            
            <div class="endpoint">
                <h2>API Endpoints:</h2>
                
                <h3>1. Process Image</h3>
                <p>Send a POST request to <code>/process</code> with the following parameters:</p>
                <ul>
                    <li><code>image</code>: The image file to process</li>
                    <li><code>box_threshold</code>: Threshold for removing bounding boxes with low confidence (default: 0.05)</li>
                    <li><code>iou_threshold</code>: Threshold for removing bounding boxes with large overlap (default: 0.1)</li>
                    <li><code>use_paddleocr</code>: Whether to use PaddleOCR (default: true)</li>
                    <li><code>imgsz</code>: Icon detect image size (default: 640)</li>
                </ul>
                <p>By default returns the processed image. If you add an Accept header with 'application/json', it will return JSON with image as base64.</p>
                
                <h4>Example with curl (get image):</h4>
                <pre>curl -X POST -F "image=@path/to/your/image.png" -F "box_threshold=0.05" -F "iou_threshold=0.1" -F "use_paddleocr=true" -F "imgsz=640" http://localhost:5000/process -o output.png</pre>
                
                <h4>Example with curl (get JSON with base64 image):</h4>
                <pre>curl -X POST -H "Accept: application/json" -F "image=@path/to/your/image.png" -F "box_threshold=0.05" -F "iou_threshold=0.1" -F "use_paddleocr=true" -F "imgsz=640" http://localhost:5000/process</pre>
            </div>
            
            <div class="endpoint">
                <h3>2. Process Image (JSON only)</h3>
                <p>Send a POST request to <code>/process_json</code> with the same parameters as above, but returns only the parsed content as JSON (no image).</p>
                
                <h4>Example with curl:</h4>
                <pre>curl -X POST -F "image=@path/to/your/image.png" -F "box_threshold=0.05" -F "iou_threshold=0.1" -F "use_paddleocr=true" -F "imgsz=640" http://localhost:5000/process_json</pre>
            </div>
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) 