# 🤖 AI Face Detector

A modern, responsive React application that uses face-api.js to perform real-time face detection, age estimation, and gender recognition directly in the browser.
https://sunny-salamander-968faf.netlify.app/
## ✨ Features

- **Real-time Face Detection**: Detect multiple faces in live video stream
- **Age Estimation**: Predict age with AI-powered accuracy
- **Gender Recognition**: Identify gender with confidence scores
- **Mobile Responsive**: Fully optimized for mobile devices
- **Modern UI**: Beautiful, intuitive interface with glassmorphism design
- **Image Capture**: Take snapshots and download them
- **Settings Panel**: Adjustable confidence thresholds
- **Dark Mode Support**: Automatic dark/light theme switching

## 🚀 Quick Start

### Prerequisites

- Node.js (version 16 or higher)
- Modern web browser with camera access
- HTTPS connection (required for camera access in production)

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd face-detector
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```

4. **Open your browser**
   Navigate to `http://localhost:5173`

## 📱 Usage

1. **Start Camera**: Click the "Start Camera" button to begin
2. **Face Detection**: The app will automatically detect faces and display bounding boxes
3. **View Results**: Age, gender, and confidence scores appear in real-time
4. **Capture Image**: Click "Capture" to take a snapshot
5. **Download**: Save captured images to your device
6. **Settings**: Adjust detection sensitivity in the settings panel

## 🛠️ Technical Details

### Core Technologies

- **React 18**: Modern React with hooks
- **Vite**: Fast build tool and dev server
- **face-api.js**: TensorFlow.js-based face detection library
- **react-webcam**: Easy webcam integration
- **Lucide React**: Beautiful icons

### AI Models Used

- **Tiny Face Detector**: Fast face detection
- **Face Landmark 68**: Facial feature detection
- **Age Gender Model**: Age and gender estimation
- **Face Expression Model**: Emotion recognition

### Browser Compatibility

- Chrome 67+
- Firefox 60+
- Safari 11+
- Edge 79+

## 📁 Project Structure

```
face-detector/
├── public/
│   └── models/          # AI model files
├── src/
│   ├── App.jsx         # Main application component
│   ├── App.css         # Styles and responsive design
│   ├── main.jsx        # Application entry point
│   └── index.css       # Global styles
├── package.json
└── README.md
```

## 🔧 Configuration

### Camera Settings

The app uses the following default camera constraints:
- Resolution: 1280x720 (ideal)
- Facing mode: User (front camera)
- Audio: Disabled

### Detection Settings

- **Confidence Threshold**: Adjustable from 10% to 90%
- **Processing Interval**: 100ms for real-time performance
- **Input Size**: 224px for optimal speed/accuracy balance

## 🚨 Important Notes

### HTTPS Requirement

Camera access requires HTTPS in production. For local development, most browsers allow camera access on `localhost`.

### Model Loading

The AI models (~2MB total) are loaded from the `/public/models/` directory. Ensure these files are accessible.

### Performance

- **Desktop**: Smooth 60fps detection
- **Mobile**: Optimized for mobile processors
- **Battery**: Efficient processing to minimize battery drain

## 🎨 Customization

### Styling

The app uses CSS custom properties and modern design patterns:
- Glassmorphism effects
- Responsive grid layouts
- Smooth animations
- Dark mode support

### Adding Features

The modular React component structure makes it easy to add:
- Additional face attributes
- Different detection algorithms
- Custom UI components
- Analytics tracking

## 🐛 Troubleshooting

### Common Issues

1. **Camera not working**
   - Check browser permissions
   - Ensure HTTPS in production
   - Try refreshing the page

2. **Models not loading**
   - Check network connection
   - Verify model files in `/public/models/`
   - Clear browser cache

3. **Poor performance**
   - Close other camera-using applications
   - Reduce browser tabs
   - Check device specifications

### Browser Support

If you encounter issues, try:
- Updating your browser
- Enabling hardware acceleration
- Disabling browser extensions

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information

---

**Built with ❤️ using React, Vite, and face-api.js**
