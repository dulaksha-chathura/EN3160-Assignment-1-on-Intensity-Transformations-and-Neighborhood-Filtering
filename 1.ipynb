import cv2
import numpy as np
import matplotlib.pyplot as plt

def intensity_transform(im, breakpoints):
    """
    Applies piece-wise linear intensity transformation on an input image.
    
    Parameters:
    im (numpy.ndarray): Grayscale input image.
    breakpoints (numpy.ndarray): Nx2 array where each row represents [input_intensity, output_intensity].
    
    Returns:
    numpy.ndarray: Transformed grayscale image.
    """
    # Sort breakpoints based on input intensities (first column)
    breakpoints = breakpoints[breakpoints[:, 0].argsort()]
    
    x = breakpoints[:, 0]
    y = breakpoints[:, 1]
    
    # Create a Look-Up Table (LUT) for intensity values 0 to 255
    # np.interp performs linear interpolation between breakpoints
    lut = np.interp(np.arange(256), x, y).astype(np.uint8)
    
    # Map the input image intensities using the LUT
    transformed_im = cv2.LUT(im, lut)
    
    return transformed_im

# ---------------------------------------------------------
# Example Execution
# ---------------------------------------------------------

# Load image in grayscale (Replace 'emma.png' with your image file path)
# If downloading from the question, ensure it is saved locally.
image_path = 'emma.png' 
im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# If reading fails (e.g., file not found), create a synthetic test image
if im is None:
    im = np.linspace(0, 255, 256*256, dtype=np.uint8).reshape(256, 256)

# Define the breakpoints matrix as specified in the problem:
# Column 0: Input intensity, Column 1: Output intensity
breakpoints = np.array([
    [0,   0],
    [50,  50],
    [50,  100],  # Jump/step discontinuity at input intensity 50
    [150, 255],
    [150, 150],  # Drop/step discontinuity at input intensity 150
    [255, 255]
], dtype=float)

# Perform transformation
transformed_im = intensity_transform(im, breakpoints)

# ---------------------------------------------------------
# Plotting Results
# ---------------------------------------------------------
plt.figure(figsize=(14, 5))

# Plot 1: Transformation Curve (Fig 1a)
plt.subplot(1, 3, 1)
plt.plot(breakpoints[:, 0], breakpoints[:, 1], 'r-', linewidth=2)
plt.title('(a) Intensity Transformation')
plt.xlabel('Input intensity')
plt.ylabel('Output intensity')
plt.xlim([0, 255])
plt.ylim([0, 255])
plt.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Original Image
plt.subplot(1, 3, 2)
plt.imshow(im, cmap='gray', vmin=0, vmax=255)
plt.title('(b) Original Image')
plt.axis('off')

# Plot 3: Transformed Image
plt.subplot(1, 3, 3)
plt.imshow(transformed_im, cmap='gray', vmin=0, vmax=255)
plt.title('Transformed Image')
plt.axis('off')

plt.tight_layout()
plt.show()
