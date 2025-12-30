# TruthLens Browser Extension

## Setup and Installation

1.  **Build the Project**:
    Run the build command to generate the distribution files.
    ```bash
    npm run build
    ```
    This will create a `dist` directory containing the compiled extension.

2.  **Load into Chrome/Brave/Edge**:
    - Open your browser and navigate to `chrome://extensions`.
    - Enable **Developer mode** (toggle in the top right corner).
    - Click **Load unpacked**.
    - Select the `browser-extension/dist` directory from this project.

## Usage

1.  Navigate to a website with a video (e.g., YouTube).
2.  Click the **TruthLens** extension icon in your toolbar.
3.  Click **Analyze Video**.
4.  The extension will start scanning the video for deepfake artifacts.
