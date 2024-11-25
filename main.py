import cv2
import matplotlib.pyplot as plt

# Function to classify shapes
def classify_shape(contour):
    # Approximate the contour
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    # Classify based on the number of vertices
    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        # Check if it's a square or rectangle
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            return "Square"
        else:
            return "Rectangle"
    elif len(approx) > 4:
        return "Circle"
    else:
        return "Unknown"

# Function to detect objects and classify shapes
def detect_shapes(image_path, selected_shape=None):
    # Load the image
    image = cv2.imread("home.jpeg")
    if image is None:
        print("Error: Image not found!")
        return

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copy the image for visualization
    image_with_contours = image.copy()

    # Detect and classify shapes
    detected_shapes = []
    for contour in contours:
        shape = classify_shape(contour)
        detected_shapes.append(shape)

        # If a specific shape is selected, only draw and label that shape
        if selected_shape is None or shape.lower() == selected_shape.lower():
            # Draw the contour
            cv2.drawContours(image_with_contours, [contour], -1, (0, 255, 0), 2)

            # Add text to label the shape
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(image_with_contours, shape, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert to RGB for displaying
    image_rgb = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)
    gray_rgb = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)  # Convert grayscale to RGB for displaying
    binary_rgb = cv2.cvtColor(binary_image, cv2.COLOR_BGR2RGB)  # Convert binary to RGB for displaying

    # Display the original, grayscale, and binary images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image with contours
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original with Contours")
    axes[0].axis("off")

    # Grayscale image
    axes[1].imshow(gray_rgb, cmap='gray')
    axes[1].set_title("Grayscale Image")
    axes[1].axis("off")

    # Binary image
    axes[2].imshow(binary_rgb, cmap='gray')
    axes[2].set_title("Binary Image")
    axes[2].axis("off")

    plt.show()

    # Print summary
    print(f"Number of objects detected: {len(contours)}")
    if selected_shape:
        count_selected = detected_shapes.count(selected_shape.capitalize())
        print(f"Number of '{selected_shape}' shapes detected: {count_selected}")

# Main program
if __name__ == "__main__":
    # Ask the user for the desired shape (optional)
    print("Shapes you can select: Triangle, Square, Rectangle, Circle")
    selected_shape = input("Enter the shape you want to detect (or press Enter to detect all objects): ").strip()

    # Path to the image
    image_path = "objects.jpg"  # Replace with your image path

    # Detect and display shapes
    detect_shapes(image_path, selected_shape if selected_shape else None)
