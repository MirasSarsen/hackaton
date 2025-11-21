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
    <div style={{ padding: 16, border: "1px solid #ddd", borderRadius: 8, background: "#fafafa" }}>
      <h3 style={{ marginTop: 0 }}>🎯 AI Media Translator</h3>

      <input type="file" onChange={(e) => setFile(e.target.files[0])}
        accept=".mp3,.wav,.mp4,.webm,.m4a,.jpg,.jpeg,.png,.webp" />

      <div style={{ marginTop: 12, display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center" }}>
        <div>
          <label>🌍 Language: </label>
          <select value={lang} onChange={(e) => setLang(e.target.value)}>
            <option value="ru">Русский</option>
            <option value="kz">Қазақша</option>
            <option value="en">English</option>
            <option value="tr">Türkçe</option>
            <option value="zh">中文</option>
          </select>
        </div>

        <div>
          <label>🎤 Voice: </label>
          <select value={voice} onChange={(e) => setVoice(e.target.value)}>
            {(VOICES[lang] || []).map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        </div>

        <div>
          <label>✨ Style: </label>
          <select value={style} onChange={(e) => setStyle(e.target.value)}>
            <option value="normal">Normal</option>
            <option value="hype">Hype 🔥</option>
            <option value="optimist">Optimist 😊</option>
            <option value="calm">Calm 🧘</option>
            <option value="formal">Formal 👔</option>
          </select>
        </div>

        <label>
          <input type="checkbox" checked={replaceImage} onChange={(e) => setReplaceImage(e.target.checked)} />
          Replace image text
        </label>

        <button onClick={upload} disabled={!!progress}
          style={{ padding: "8px 16px", background: "#4CAF50", color: "white", border: "none", borderRadius: 4, cursor: "pointer" }}>
          {progress || "Upload & Process"}
        </button>
      </div>
    </div>
  );
}
