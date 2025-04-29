import React, { useState, useRef } from 'react';
import ImageUploader from './ImageUploader';

import coralReefImage from './co.jpg'; // Import the image

function App() {
  const [image, setImage] = useState(null);
  const [webcamStream, setWebcamStream] = useState(null);
  const videoRef = useRef(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImage(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      setWebcamStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error("Error accessing webcam:", error);
      alert("Could not access webcam. Please ensure you have given permission.");
    }
  };

  const handleStopWebcam = () => {
    if (webcamStream) {
      webcamStream.getTracks().forEach(track => track.stop());
      setWebcamStream(null);
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    }
  };

  const openModal = (type) => {
    setModalType(type);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setModalType('');
  };
  return (
    <div className="app-container">
      <header className="header">
        <h1 >Coral Reef Health Detection</h1>
        <div className="header-buttons">
          <button className="login-button" onClick={() => openModal('Login')}>Login</button>
          <button className="register-button" onClick={() => openModal('Register')}>Register</button>
          <button className="contact-button">Contact Us</button>
        </div>
      </header>

      <main className="main-content">
        <div className="App">
              <ImageUploader />
        </div>
        <section className="details-section">
          <h1 style={{textAlign: 'center', marginBottom: '35px', fontSize:'1.8rem ', marginTop:'5rem'}}>Protecting Coral Reefs With Technology</h1>
          <div className="details-content">

            <div className="image-container" style={{marginLeft:'40px'}}>
              <img src={coralReefImage} alt="Coral Reef" className="coral-image" />
            </div>
            <div className="text-container">
              <h2>Why Coral Reef Health Matters</h2>
              <p>
                Coral reefs are among the most valuable ecosystems on Earth, supporting approximately 25% of marine life while covering less than 1% of the ocean floor. They provide food security, coastal protection, and economic benefits to millions of people.
                Yet these vital ecosystems face unprecedented threats from climate change, pollution, and unsustainable fishing practices. Monitoring coral reef health is essential for effective conservation and management.
              </p>
              <p>
                The analysis may include identifying signs of coral bleaching, disease, or damage. The goal is to provide a simple and accessible tool
                for anyone interested in coral reef conservation.
              </p> 
              </div>
              </div>

              <div className="technology">
              <h3 style={{fontSize:'1.5rem'}}>Our Technology</h3>
              <h5 style={{fontSize:'1rem'}}>Global Reef Monitoring:</h5>
              <p>
                Track coral reef health across the globe with our comprehensive database of reef ecosystems and historical health trends.
              </p>
              <h5 style={{fontSize:'1rem'}}>Conservation Insights:</h5>
              <p>
                Gain actionable insights for coral reef conservation efforts with detailed reports and recommendations based on analysis results.
              </p>
            </div>
          
        </section>
      </main>
 
      <footer className="footer">
        <p>&copy; 2024 Coral Reef Health Detection</p>
      </footer>

      {/*{isModalOpen && (
        <div className="modal">
          <div className="modal-content">
            <h2>{modalTy
            pe} with Google</h2>
            <button className="google-button">Sign in with Google</button>
            <button className="close-button" onClick={closeModal}>Close</button>
          </div>
        </div>
      )}*/}
    </div>
  );
}

export default App;