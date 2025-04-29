import React, { useState } from "react";
import axios from "axios";

const ImageUploader = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [model, setModel] = useState("cnn"); // Default to CNN
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null); // Confidence level for the prediction
  const [isLoading, setIsLoading] = useState(false); // Loading state for the "Find" button

  // Handle file selection
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile)); // Generate a preview URL for the image
      setPrediction(null); // Reset prediction when a new file is selected
    }
  };

  // Handle model selection
  const handleModelChange = (event) => {
    setModel(event.target.value);
    setPrediction(null); // Reset prediction when a new model is selected
  };

  // Handle file upload and send model choice

  const handleFind = async () => {
    if (!file) {
      alert("Please select an image first!");
      return;
    }

    setIsLoading(true); // Show loading state
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", model); // Send model choice to backend

    try {
      const response = await axios.post("http://localhost:8000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPrediction(response.data.result); // Update prediction with the result from the backend
      setConfidence(response.data.confidence); // Update confidence level
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("An error occurred while processing the image. Please try again.");
    } finally {
      setIsLoading(false); // Hide loading state
    }
  };


  return (
    <section className="image-upload-section">
    <div>
      <h2>Upload an Image</h2>
      <div className="upload-container">
      <input
      type="file"
      id="file-upload" // Add an id to associate with the label
      onChange={handleFileChange}
      style={{ display: "none" }} // Hide the input element
      />
      {!file && ( // Conditionally render the upload button if no file is selected
            <label htmlFor="file-upload" className="upload-button">
              Upload Image
            </label>
          )}
      {preview && (
          <div className="image-preview">
            <h3>Selected Image:</h3>
            <img src={preview} alt="Preview" className="preview-image" />
          </div>
        )}
      </div>
        
      {file && ( // Render the model selection and "Find" button only after an image is selected
          <div className="model-selection-container">
          <label>Select Model:</label>
          <select value={model} onChange={handleModelChange} className="model-dropdown">
            <option value="cnn">CNN Model</option>
            <option value="resnet">ResNet Model</option>
            <option value="densenet">DenseNet Model</option>
            <option value="efficientnet">Efficient Net Model</option>
          </select>
        
          <button onClick={handleFind} disabled={isLoading} className="find-button">
            {isLoading ? "Processing..." : "Find"}
          </button>
        </div>
        )}

      {prediction && (
        <div className="prediction-result">
          <h3>Prediction: {prediction}</h3>
          {confidence !== null && <p>Confidence : {confidence}%</p>}
        </div>
        )}
    </div>
    </section>
  );
};

export default ImageUploader;