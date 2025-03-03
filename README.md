# Stable Diffusion with Concepts & Guidance

This project demonstrates how to generate images using Stable Diffusion v1-4, incorporating custom concepts via textual inversion and an optional forced guidance technique.  It's designed to run on CPU (though it can be adapted for GPU) and provides a Gradio interface for easy experimentation.

## Technologies Used

*   **PyTorch:**  A deep learning framework used for tensor operations and model execution.
*   **Diffusers:** A library by Hugging Face that provides pre-built components and pipelines for diffusion models, simplifying the implementation of Stable Diffusion.
*   **Transformers:** Another library by Hugging Face, essential for the text encoding part of Stable Diffusion.  It provides the pre-trained text encoder (usually CLIP) used to convert text prompts into embeddings.

## Stable Diffusion Process

Stable Diffusion is a latent diffusion model.  Here's a breakdown of the process:

1.  **Text Encoding:**
    *   The user provides a text prompt (e.g., "A cat wearing a hat").
    *   The `tokenizer` converts this prompt into token IDs.
    *   The `text_encoder` (usually a pre-trained CLIP model) transforms these token IDs into a text embedding, a high-dimensional vector representing the semantic meaning of the prompt.

2.  **Latent Space:**
    *   Stable Diffusion operates in a "latent space," a compressed representation of images created by the Variational Autoencoder (VAE). This reduces computational requirements.
    *   Random noise is generated in this latent space.

3.  **Denoising (Reverse Diffusion):**
    *   The core of Stable Diffusion is a UNet model that iteratively removes noise from the latent representation.
    *   The UNet takes the noisy latent, the timestep, and the text embedding as input.
    *   It predicts the noise that was added to the latent.
    *   The scheduler uses this noise prediction to step backward, gradually refining the latent towards a coherent image representation.

4.  **VAE Decoding:**
    *   Once the denoising process is complete, the final latent representation is decoded back into an image by the VAE decoder.

5.  **Safety Checker:**
    *   A safety checker is used to detect and blur potentially unsafe or inappropriate content in the generated image.

## SD Concepts (Textual Inversion)

This project incorporates custom concepts into the Stable Diffusion process using a technique called "Textual Inversion."

*   **What is Textual Inversion?**  It's a method for teaching Stable Diffusion new visual concepts without retraining the entire model.  It involves finding a new "word" (a token embedding) in the text encoder's vocabulary that, when used in a prompt, causes the model to generate images containing the desired concept.

*   **How it Works:**
    1.  A set of images representing the new concept (e.g., a specific art style, a character) is used to train a new token embedding.
    2.  The training process optimizes this embedding so that when the corresponding token is used in a prompt, the generated image reflects the concept.
    3.  The learned embedding is stored in a `.bin` file (e.g., `my-concept-learned_embeds.bin`).

*   **Implementation in this Project:**
    *   The code loads pre-trained concept embeddings from `.bin` files.
    *   It uses a placeholder token (e.g., `<my_concept>`) in the prompt.
    *   During the image generation process, the embedding for the placeholder token is replaced with the loaded concept embedding.

## Forced Guidance (Creative Loss)

This project implements a form of "forced guidance" to influence the image generation process. In this case, it uses a cosine similarity loss to guide the image generation towards a specific visual characteristic (airplanes).

*   **What is Forced Guidance?** It's a technique to steer the diffusion process towards a desired outcome by calculating a loss based on the current latent representation and a target representation, and then using the gradient of this loss to adjust the latent.

*   **How it Works:**
    1.  **Target Latent Extraction:** A set of images representing the desired characteristic (airplanes) is used to extract an average latent embedding using the VAE encoder.
    2.  **Loss Calculation:** During the denoising loop, the cosine similarity between the current latent representation and the target latent embedding is calculated. This measures how closely the current latent resembles the desired characteristic.
    3.  **Gradient Descent:** The gradient of the cosine similarity loss is calculated with respect to the current latent. This gradient indicates the direction in which the latent needs to be adjusted to increase its similarity to the target.
    4.  **Latent Adjustment:** The current latent is adjusted by subtracting the gradient, scaled by a guidance factor. This steers the denoising process towards generating images with the desired characteristic.

*   **Implementation in this Project:**
    *   The code extracts an average latent embedding from a set of airplane images.
    *   During the denoising loop, it calculates the cosine similarity between the current latent and the airplane latent.
    *   It uses the gradient of this loss to adjust the latent, encouraging the generated image to resemble an airplane.

## Project Structure
```
SD-With-Concepts-Guidance/ 
├── app.py # Main application file with Gradio interface 
├── sd-concepts/ # Directory for storing concept embeddings (.bin files) 
│ ├── <concept1>-learned_embeds.bin 
│ ├── <concept2>-learned_embeds.bin 
│ └── ... 
├── clip_images/ # Directory for storing airplane images 
│ ├── plane_0.png 
│ ├── plane_1.png 
│ └── ... 
├── README.md # This file 
└── requirements.txt # List of Python dependencies
```

## Setup and Usage

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Concept Embeddings:**

    *   Download the learned embeddings (`.bin` files) for the concepts you want to use.  You can find these in community-created SD concepts libraries (e.g., on Hugging Face).
    *   Place the `.bin` files in the [sd-concepts](http://_vscodecontentref_/1) directory.  Rename them according to the convention `<concept_name>-learned_embeds.bin`.

3.  **Prepare Airplane Images (Optional):**

    *   If you want to use the airplane guidance, generate or download a set of airplane images.
    *   Place the images in the [clip_images](http://_vscodecontentref_/2) directory, named `plane_0.png`, `plane_1.png`, etc.

4.  **Run the Application:**

    ```bash
    python app.py
    ```

5.  **Access the Gradio Interface:**

    *   Open your web browser and go to the address displayed in the console (usually `http://localhost:7860`).

6.  **Experiment:**

    *   Enter a prompt in the "Prompt Base" textbox.
    *   Select a concept token from the "Concept Token" dropdown.
    *   Adjust the "Seed" slider to generate different images.
    *   Adjust the "Airplane Guidance Scale" slider to control the strength of the airplane guidance.  A value of 0 disables the guidance.

## Customization

*   **Adding New Concepts:**
    1.  Train a new concept embedding using textual inversion techniques.
    2.  Place the `.bin` file in the [sd-concepts](http://_vscodecontentref_/3) directory.
    3.  Add the corresponding token to the [concepts](http://_vscodecontentref_/4) list in [app.py](http://_vscodecontentref_/5).
    4.  Update the Gradio interface to include the new concept.

*   **Modifying the Guidance:**
    *   You can experiment with different loss functions for the forced guidance.
    *   You can use different target representations (e.g., latent embeddings from other types of images).
    *   You can adjust the guidance scale to control the strength of the guidance.

## Important Notes

*   **CPU Performance:**  Running Stable Diffusion on CPU can be slow.  Consider using a GPU for faster generation.
*   **Memory Usage:**  Stable Diffusion requires a significant amount of memory.  Reduce the image size or batch size if you encounter memory issues.
*   **Hugging Face Token:**  Make sure you have a valid Hugging Face token and have authenticated with `huggingface-cli login` if you are using models that require authentication.
*   **Concept Embeddings:** The quality of the generated images depends heavily on the quality of the concept embeddings.

This README provides a comprehensive overview of the project.  Feel free to experiment and modify the code to explore the capabilities of Stable Diffusion!
