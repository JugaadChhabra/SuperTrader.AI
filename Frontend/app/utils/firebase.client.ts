// src/firebase.client.ts
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

// Your web app's Firebase configuration (from your Firebase Project settings)
const firebaseConfig = {
  apiKey: "AIzaSyBUyTjZId7Qfiye8xRe825RcuE5v3nIeYA",
  authDomain: "capstone-ff4c0.firebaseapp.com",
  projectId: "capstone-ff4c0",
  storageBucket: "capstone-ff4c0.firebasestorage.app",
  messagingSenderId: "511218534745",
  appId: "1:511218534745:web:8b40af90c602bea272a448",
  measurementId: "G-CLMMQJBN05"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase Authentication and get a reference to the service
const auth = getAuth(app);

export { auth, app };
