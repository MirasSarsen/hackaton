import React from "react";
import axios from "axios";

const VOICES = {
  ru: ["aidar", "baya", "kseniya", "xenia", "eugene", "random"],
  en: ["en_0", "en_1", "en_2", "en_3", "en_4", "random"],
  de: ["bernd_ungerer", "eva_k", "friedrich", "karlsson", "random"],
  es: ["es_0", "es_1", "es_2", "random"],
  fr: ["fr_0", "fr_1", "fr_2", "fr_3", "fr_4", "random"],
  kz: ["en_0", "en_1", "random"],
  tr: ["en_0", "en_1", "random"],
  zh: ["en_0", "en_1", "random"],
};

export default function UploadForm({ onResult }) {
  const [file, setFile] = React.useState(null);
  const [lang, setLang] = React.useState("ru");
  const [voice, setVoice] = React.useState("random");
  const [style, setStyle] = React.useState("normal");
  const [replaceImage, setReplaceImage] = React.useState(true);
  const [progress, setProgress] = React.useState(null);
  const [ttsLoading, setTtsLoading] = React.useState(false);

  // Update voice when language changes
  React.useEffect(() => {
    setVoice(VOICES[lang]?.[0] || "random");
  }, [lang]);

  const upload = async () => {
    if (!file) return alert("Choose file");

    const fd = new FormData();
    fd.append("file", file);

    setProgress("uploading...");
    const up = await axios.post("http://localhost:8000/upload", fd);
    const filename = up.data.filename;

    const form = new FormData();
    form.append("filename", filename);
    form.append("translate_to", lang);
    form.append("style", style);
    form.append("replace_image_text", replaceImage ? "true" : "false");

    setProgress("processing...");
    const proc = await axios.post("http://localhost:8000/process", form);

    onResult({ ...proc.data, selectedLang: lang, selectedVoice: voice });
    setProgress(null);
  };

  return (
    <div className="card upload-form">
      <h3>🎯 AI Media Translator</h3>

      <div className="file-input-container">
        <input 
          type="file" 
          id="file-input"
          className="file-input"
          onChange={(e) => setFile(e.target.files[0])}
          accept=".mp3,.wav,.mp4,.webm,.m4a,.jpg,.jpeg,.png,.webp" 
        />
        <label 
          htmlFor="file-input" 
          className={`file-input-label ${file ? 'file-selected' : ''}`}
        >
          {file ? (
            <>
              ✅ {file.name}
              <div style={{ fontSize: '0.9rem', marginTop: '5px', opacity: 0.8 }}>
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </div>
            </>
          ) : (
            <>
              📁 Choose audio, video, or image file
              <div style={{ fontSize: '0.9rem', marginTop: '5px', opacity: 0.8 }}>
                Supports: MP3, WAV, MP4, WebM, M4A, JPG, PNG, WebP
              </div>
            </>
          )}
        </label>
      </div>

      <div className="controls-grid">
        <div className="control-group">
          <label className="control-label">🌍 Target Language</label>
          <select 
            value={lang} 
            onChange={(e) => setLang(e.target.value)}
            className="select-input"
          >
            <option value="ru">🇷🇺 Русский</option>
            <option value="kz">🇰🇿 Қазақша</option>
            <option value="en">🇺🇸 English</option>
            <option value="tr">🇹🇷 Türkçe</option>
            <option value="zh">🇨🇳 中文</option>
          </select>
        </div>

        <div className="control-group">
          <label className="control-label">🎤 Voice Style</label>
          <select 
            value={voice} 
            onChange={(e) => setVoice(e.target.value)}
            className="select-input"
          >
            {(VOICES[lang] || []).map(v => (
              <option key={v} value={v}>
                {v === 'random' ? '🎲 Random' : v}
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label className="control-label">✨ Paraphrase Style</label>
          <select 
            value={style} 
            onChange={(e) => setStyle(e.target.value)}
            className="select-input"
          >
            <option value="normal">📝 Normal</option>
            <option value="hype">🔥 Hype</option>
            <option value="optimist">😊 Optimist</option>
            <option value="calm">🧘 Calm</option>
            <option value="formal">👔 Formal</option>
          </select>
        </div>
      </div>

      <div className="checkbox-container">
        <input 
          type="checkbox" 
          id="replace-image"
          className="checkbox-input"
          checked={replaceImage} 
          onChange={(e) => setReplaceImage(e.target.checked)} 
        />
        <label htmlFor="replace-image" className="checkbox-label">
          🖼️ Replace text in images with translations
        </label>
      </div>

      <button 
        onClick={upload} 
        disabled={!!progress || !file}
        className="btn btn-primary"
        style={{ width: '100%' }}
      >
        {progress ? (
          <div className="progress-indicator">
            <div className="spinner"></div>
            {progress}
          </div>
        ) : (
          <>🚀 Upload & Process</>
        )}
      </button>
    </div>
  );
}
