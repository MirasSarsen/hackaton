import React from "react";
import UploadForm from "./components/UploadForm";
import ResultView from "./components/ResultView";
import "./styles.css";

export default function App(){
  const [result, setResult] = React.useState(null);
  
  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">🌍 AI Media Translator</h1>
        <p className="app-subtitle">
          Translate audio, video, and images with AI-powered speech recognition and text-to-speech
        </p>
      </header>
      
      <main>
        <UploadForm onResult={setResult} />
        {result && <ResultView result={result} />}
      </main>
    </div>
  );
}
