import React, { useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import image2 from './Dubai_Marina_Skyline.jpg';
import image3 from './forest.webp';
import image4 from './sea.jpg';

const ObjectDetection = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const runObjectDetection = async () => {
      // Load your custom TensorFlow.js model
      console.log('Custom model going to load.');
      const model = await tf.loadLayersModel('https://raw.githubusercontent.com/VarunCypherV/ObjectDetectionReactApp/main/model.json');
      console.log('Custom model loaded.');
      console.log(model);

      // Load the image for prediction
      const img = new Image();
      img.src = image4;
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

        // Define the class labels
        const classLabels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'];

        // Find the index with the highest probability
        const maxIndex = predictions.indexOf(Math.max(...predictions));

        // Log the predicted class
        console.log('Predicted Class:', classLabels[maxIndex]);
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
