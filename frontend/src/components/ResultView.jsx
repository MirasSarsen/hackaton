import React from "react";
import axios from "axios";

export default function ResultView({ result }) {
  const [ttsAudio, setTtsAudio] = React.useState(null);
  const [ttsLoading, setTtsLoading] = React.useState(false);

  const generateTTS = async (text, type) => {
    setTtsLoading(true);
    try {
      const form = new FormData();
      form.append("text", text.slice(0, 1000));
      form.append("lang", result.selectedLang || "en");
      form.append("speaker", result.selectedVoice || "random");

      const res = await axios.post("http://localhost:8000/tts", form, { responseType: "blob" });
      const url = URL.createObjectURL(res.data);
      setTtsAudio({ url, type });
    } catch (e) {
      alert("TTS error: " + e.message);
    }
    setTtsLoading(false);
  };

  const formatTime = (sec) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="card results-container">
      <h3 className="results-title">📊 Processing Results</h3>

      {/* Original Text */}
      <div className="result-section original-section">
        <div className="result-section-title">
          <span>📝 Original Text</span>
          <button 
            onClick={() => generateTTS(result.original_text, "original")}
            disabled={ttsLoading} 
            className="btn btn-secondary"
          >
            🔊 Listen
          </button>
        </div>
        <div className="result-text">
          {result.original_text}
        </div>
      </div>

      {/* Speaker Diarization */}
      {result.transcription?.segments && (
        <div className="result-section transcription-container">
          <div className="result-section-title">
            <span>🎙️ Transcription with Speakers & Timestamps</span>
          </div>
          
          {result.transcription.language && (
            <div className="transcription-meta">
              <span>Detected language: <strong>{result.transcription.language}</strong></span>
              <span>Duration: <strong>{formatTime(result.transcription.duration || 0)}</strong></span>
            </div>
          )}
          
          {result.transcription.segments.map((seg, i) => (
            <div key={i} className={`speaker-segment ${seg.speaker === "Speaker 1" ? "speaker-1" : "speaker-2"}`}>
              <div className="speaker-header">
                <span className="speaker-name">{seg.speaker}</span>
                <span className="timestamp">
                  ⏱️ {formatTime(seg.start)} - {formatTime(seg.end)}
                </span>
              </div>
              <div>{seg.text}</div>
              {seg.words?.length > 0 && (
                <details className="words-details">
                  <summary>Show words ({seg.words.length})</summary>
                  <div className="words-container">
                    {seg.words.map((w, j) => (
                      <span key={j} className="word-chip">
                        {w.word} <span style={{ color: "#aaa" }}>({formatTime(w.start)})</span>
                      </span>
                    ))}
                  </div>
                </details>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Translation */}
      <div className="result-section translation-section">
        <div className="result-section-title">
          <span>🌐 Translation</span>
          <button 
            onClick={() => generateTTS(result.translation, "translation")}
            disabled={ttsLoading} 
            className="btn btn-secondary"
          >
            🔊 Listen
          </button>
        </div>
        <div className="result-text">
          {result.translation}
        </div>
      </div>

      {/* Paraphrase */}
      <div className="result-section paraphrase-section">
        <div className="result-section-title">
          <span>✨ Paraphrase</span>
          <button 
            onClick={() => generateTTS(result.paraphrase, "paraphrase")}
            disabled={ttsLoading} 
            className="btn btn-secondary"
          >
            🔊 Listen
          </button>
        </div>
        <div className="result-text">
          {result.paraphrase}
        </div>
      </div>

      {/* TTS Audio Player */}
      {ttsAudio && (
        <div className="result-section audio-container">
          <div className="result-section-title">
            <span>🎵 Generated Audio ({ttsAudio.type})</span>
          </div>
          <audio controls src={ttsAudio.url} className="audio-player" />
        </div>
      )}

      {/* OCR Lines */}
      {result.ocr_lines && (
        <details style={{ marginBottom: 16 }}>
          <summary><strong>📦 OCR Boxes ({result.ocr_lines.length})</strong></summary>
          <pre style={{ fontSize: 11 }}>{JSON.stringify(result.ocr_lines, null, 2)}</pre>
        </details>
      )}

      {/* Translated Image */}
      {result.translated_image_url && (
        <div className="result-section">
          <div className="result-section-title">
            <span>🖼️ Translated Image</span>
          </div>
          <img 
            src={`http://localhost:8000${result.translated_image_url}`} 
            alt="Translated image with replaced text"
            className="translated-image" 
          />
        </div>
      )}

      {/* Download Links */}
      <div className="download-section">
        <div className="result-section-title">
          <span>📥 Download Files</span>
        </div>
        <div className="download-links">
          <a href={`http://localhost:8000${result.files.translation}`} download className="download-link">
            📄 Translation.txt
          </a>
          <a href={`http://localhost:8000${result.files.paraphrase}`} download className="download-link">
            ✨ Paraphrase.txt
          </a>
        </div>
      </div>
    </div>
  );
}
