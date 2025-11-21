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
    <div style={{ marginTop: 16, padding: 16, border: "1px solid #eee", borderRadius: 8 }}>
      <h3>📊 Results</h3>

      {/* Original Text */}
      <div style={{ marginBottom: 16 }}>
        <strong>📝 Original Text:</strong>
        <button onClick={() => generateTTS(result.original_text, "original")}
          disabled={ttsLoading} style={{ marginLeft: 8, fontSize: 12 }}>
          🔊 Listen
        </button>
        <pre style={{ whiteSpace: "pre-wrap", background: "#f5f5f5", padding: 8, borderRadius: 4 }}>
          {result.original_text}
        </pre>
      </div>

      {/* Speaker Diarization - NEW! */}
      {result.transcription?.segments && (
        <div style={{ marginBottom: 16 }}>
          <strong>🎙️ Transcription with Speakers & Timestamps:</strong>
          <div style={{ background: "#f0f8ff", padding: 12, borderRadius: 4, marginTop: 8 }}>
            {result.transcription.language && (
              <div style={{ marginBottom: 8, color: "#666" }}>
                Detected language: <b>{result.transcription.language}</b> |
                Duration: <b>{formatTime(result.transcription.duration || 0)}</b>
              </div>
            )}
            {result.transcription.segments.map((seg, i) => (
              <div key={i} style={{
                marginBottom: 8,
                padding: 8,
                background: seg.speaker === "Speaker 1" ? "#e3f2fd" : "#fff3e0",
                borderRadius: 4,
                borderLeft: `4px solid ${seg.speaker === "Speaker 1" ? "#2196F3" : "#FF9800"}`
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontWeight: "bold", color: seg.speaker === "Speaker 1" ? "#1976D2" : "#F57C00" }}>
                    {seg.speaker}
                  </span>
                  <span style={{ color: "#666", fontSize: 12 }}>
                    ⏱️ {formatTime(seg.start)} - {formatTime(seg.end)}
                  </span>
                </div>
                <div>{seg.text}</div>
                {seg.words?.length > 0 && (
                  <details style={{ marginTop: 4, fontSize: 11, color: "#888" }}>
                    <summary>Show words ({seg.words.length})</summary>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
                      {seg.words.map((w, j) => (
                        <span key={j} style={{ background: "#fff", padding: "2px 4px", borderRadius: 2 }}>
                          {w.word} <span style={{ color: "#aaa" }}>({formatTime(w.start)})</span>
                        </span>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Translation */}
      <div style={{ marginBottom: 16 }}>
        <strong>🌐 Translation:</strong>
        <button onClick={() => generateTTS(result.translation, "translation")}
          disabled={ttsLoading} style={{ marginLeft: 8, fontSize: 12 }}>
          🔊 Listen
        </button>
        <pre style={{ whiteSpace: "pre-wrap", background: "#e8f5e9", padding: 8, borderRadius: 4 }}>
          {result.translation}
        </pre>
      </div>

      {/* Paraphrase */}
      <div style={{ marginBottom: 16 }}>
        <strong>✨ Paraphrase:</strong>
        <button onClick={() => generateTTS(result.paraphrase, "paraphrase")}
          disabled={ttsLoading} style={{ marginLeft: 8, fontSize: 12 }}>
          🔊 Listen
        </button>
        <pre style={{ whiteSpace: "pre-wrap", background: "#fff3e0", padding: 8, borderRadius: 4 }}>
          {result.paraphrase}
        </pre>
      </div>

      {/* TTS Audio Player */}
      {ttsAudio && (
        <div style={{ marginBottom: 16, padding: 12, background: "#e1f5fe", borderRadius: 4 }}>
          <strong>🎵 Generated Audio ({ttsAudio.type}):</strong>
          <audio controls src={ttsAudio.url} style={{ width: "100%", marginTop: 8 }} />
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
        <div style={{ marginBottom: 16 }}>
          <strong>🖼️ Translated Image:</strong>
          <img src={`http://localhost:8000${result.translated_image_url}`} alt="translated"
            style={{ maxWidth: "100%", border: "1px solid #ddd", marginTop: 8, borderRadius: 4 }} />
        </div>
      )}

      {/* Download Links */}
      <div style={{ marginTop: 16 }}>
        <strong>📥 Download:</strong>
        <a href={`http://localhost:8000${result.files.translation}`} download style={{ marginLeft: 8 }}>
          Translation.txt
        </a>
        <a href={`http://localhost:8000${result.files.paraphrase}`} download style={{ marginLeft: 12 }}>
          Paraphrase.txt
        </a>
      </div>
    </div>
  );
}

// App.jsx
export function App() {
  const [result, setResult] = React.useState(null);

  return (
    <div style={{ maxWidth: 900, margin: "20px auto", fontFamily: "system-ui" }}>
      <UploadForm onResult={setResult} />
      {result && <ResultView result={result} />}
    </div>
  );
}
