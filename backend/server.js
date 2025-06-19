/**
 * Quickcap Backend Server
 * 
 * This server handles backend functionality for the Quickcap application.
 */

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const dotenv = require('dotenv');
const admin = require('firebase-admin');


// Load environment variables
dotenv.config();

// Initialize Firebase Admin SDK
if (!admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert({
      projectId: process.env.FIREBASE_PROJECT_ID,
      clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
      privateKey: process.env.FIREBASE_PRIVATE_KEY.replace(/\\n/g, '\n'),
    }),
    databaseURL: process.env.FIREBASE_DATABASE_URL,
  });
}

// Create Express app
const app = express();

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Add Content Security Policy headers
app.use((req, res, next) => {
  res.setHeader(
    'Content-Security-Policy',
    "frame-ancestors 'self' https://embed.elephant.ai https://bot.elephant.ai;"
  );
  
  // Add Permissions Policy for microphone
  res.setHeader(
    'Permissions-Policy',
    'camera=(), microphone=(self "https://bot.elephant.ai" "https://embed.elephant.ai"), geolocation=()'
  );
  
  next();
});

// Routes

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.status(200).json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({
    error: 'Internal Server Error',
    message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong',
  });
});

// Start server
const PORT = process.env.NODE_PORT || 3001;
const HOST = process.env.NODE_HOST || '0.0.0.0';
app.listen(PORT, HOST, () => {
  console.log(`Node.js server running on ${HOST}:${PORT}`);
});

module.exports = app;