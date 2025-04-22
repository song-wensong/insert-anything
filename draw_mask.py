import gradio as gr
import numpy as np
from PIL import Image

def process_mask(mask_input):
    # mask_input is expected to be a dictionary {'image': np.array, 'mask': np.array}
    # or just {'mask': np.array} if no image was uploaded initially.
    # Handle potential None mask if nothing is drawn or cleared
    if mask_input is None or "mask" not in mask_input or mask_input["mask"] is None:
        print("Warning: No mask data found.")
        # Return a blank image or indicate no mask found visually
        # Creating a small transparent placeholder might be better than None
        # return np.zeros((100, 100, 4), dtype=np.uint8) # Example placeholder
        return None # Keep returning None for simplicity unless placeholder needed

    mask_array = mask_input["mask"]

    try:
        # Save the mask (overwrites existing file)
        # Ensure mask is saved in a usable format (e.g., grayscale)
        mask_image = Image.fromarray(mask_array).convert("L") # Convert to grayscale
        mask_image.save("mask.png")
        print("Mask saved to mask.png")
    except Exception as e:
        print(f"Error saving mask: {e}")
        # Decide if you still want to return the mask for display
        # return None # Option: Return None on save error

    # Return the mask array for display in the output component
    # The output component will display it (might look like a black/white image)
    return mask_array

with gr.Blocks() as demo:
    gr.Markdown("### Draw Mask\nUpload an image, then use the sketch tool to create a mask. The display area will resize to fit the image.")

    with gr.Row():
        mask_editor = gr.Image(
            label="Draw Mask Here",
            source="upload",
            tool="sketch",
            type="numpy",
            interactive=True,
        )

        output_mask = gr.Image(
            label="Generated Mask Preview",
            type="numpy",
            interactive=False,
        )

    btn = gr.Button("Process and Save Mask")

    btn.click(
        fn=process_mask,
        inputs=[mask_editor],
        outputs=[output_mask]
    )

if __name__ == "__main__":
    demo.launch()