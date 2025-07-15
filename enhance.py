import os
import torch
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Import your model and metrics
from major_final import MLDRG, calculate_psnr, calculate_ssim, calculate_uiqm, calculate_uciqe

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pretrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLDRG().to(device)
model.load_state_dict(torch.load("mldrg_hfl.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_file = request.files['raw']
        if image_file:
            filename = secure_filename(image_file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(path)

            # Load and preprocess image
            img = Image.open(path).convert("RGB").resize((256, 256))
            input_tensor = transform(img).unsqueeze(0).to(device)

            # Enhance
            with torch.no_grad():
                enhanced_tensor = model(input_tensor).squeeze(0).clamp(0, 1).cpu()

            # Save enhanced image
            enhanced_filename = "enhanced_" + filename
            enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], enhanced_filename)
            save_image(enhanced_tensor, enhanced_path)

            # Compute metrics
            original = input_tensor.squeeze(0).cpu()
            psnr_val = calculate_psnr(original, enhanced_tensor)
            ssim_val = calculate_ssim(original, enhanced_tensor)
            uiqm_val = calculate_uiqm(enhanced_tensor)
            uciqe_val = calculate_uciqe(enhanced_tensor)

            return render_template('index.html',
                                   original_img=f"uploads/{filename}",
                                   enhanced_img=f"uploads/{enhanced_filename}",
                                   psnr=psnr_val,
                                   ssim=ssim_val,
                                   uiqm=uiqm_val,
                                   uciqe=uciqe_val)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
# This code is a Flask web application that allows users to upload an image, enhance it using a pre-trained MLDRG model, and display the results along with various image quality metrics.
# The application uses PyTorch for model inference and PIL for image processing. The enhanced image and metrics are displayed on the web page after processing.