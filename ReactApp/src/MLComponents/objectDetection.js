import React, { useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import image from './20066.jpg';
import image1 from './20191.jpg';
const ObjectDetection = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const runObjectDetection = async () => {
      // Load your custom TensorFlow.js model
      console.log('Custom model going to load.');
      const model = await tf.loadLayersModel('https://raw.githubusercontent.com/VarunCypherV/ObjectDetectionReactApp/main/model.json');
      console.log('Custom model loaded.');

      // Load the image for prediction
      const img = new Image();
      img.src = image1;
      console.log("going in baby");
      img.onload = async () => {
        // Ensure the image has the desired dimensions (256x256)
        console.log("i am in baby");
        const resizedImage = tf.image.resizeBilinear(tf.browser.fromPixels(img), [256, 256]);

        // Normalize the pixel values to be between 0 and 1
        const normalizedImage = resizedImage.div(255.0);

        // Expand dimensions to match the model's input shape
        const inputTensor = normalizedImage.expandDims(0);
        
        // Make predictions
        const predictions = await model.predict(inputTensor).data();
        
        // Log the predictions
        console.log('Predictions:', predictions);
        
        // Here you can use the predictions to draw on the canvas or perform other actions.
        // For drawing, you might need to set up your canvas and draw boxes based on the predictions.
      };
    };

    runObjectDetection();
  }, []);

  return (
    <div>
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          margin: "auto",
          textAlign: "center",
          zIndex: 9,
          height: 800,
          width: 800,
        }}
      />
    </div>
  );
};

export default ObjectDetection;
