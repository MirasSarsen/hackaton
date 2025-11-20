import React from "react";
import axios from "axios";

export default function UploadForm({ onResult }) {
  const [file, setFile] = React.useState(null);
  const [lang, setLang] = React.useState("ru");
  const [style, setStyle] = React.useState("normal");
  const [replaceImage, setReplaceImage] = React.useState(false);
  const [progress, setProgress] = React.useState(null);

  const upload = async () => {
    if (!file) return alert("Choose file");

    // upload file
    const fd = new FormData();
    fd.append("file", file);

    setProgress("uploading");
    const up = await axios.post("http://localhost:8000/upload", fd);

    const filename = up.data.filename;
    setProgress("uploaded");

    // second request
    const form = new FormData();
    form.append("filename", filename);
    form.append("translate_to", lang);
    form.append("style", style);
    form.append("replace_image_text", replaceImage ? "true" : "false");

    setProgress("processing");
    const proc = await axios.post("http://localhost:8000/process", form);

    onResult(proc.data);
    setProgress(null);
  };

  return (
    <div style={{ padding: 12, border: "1px solid #ddd", borderRadius: 8 }}>
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />

      <div style={{ marginTop: 8 }}>
        <label>Translate to: </label>
        <select value={lang} onChange={(e) => setLang(e.target.value)}>
          <option value="ru">Русский</option>
          <option value="kz">Қазақша</option>
          <option value="en">English</option>
          <option value="tr">Turkish</option>
          <option value="zh">Chinese</option>
        </select>

        <label style={{ marginLeft: 12 }}>Style: </label>
        <select value={style} onChange={(e) => setStyle(e.target.value)}>
          <option value="normal">Normal</option>
          <option value="hype">Hype</option>
          <option value="optimist">Optimist</option>
          <option value="calm">Calm</option>
        </select>

        <label style={{ marginLeft: 12 }}>
          <input
            type="checkbox"
            checked={replaceImage}
            onChange={(e) => setReplaceImage(e.target.checked)}
          />
          Replace image text
        </label>

        <button style={{ marginLeft: 12 }} onClick={upload}>
          Upload & Process
        </button>
      </div>

      {progress && <div style={{ marginTop: 8 }}>Status: {progress}</div>}
    </div>
  );
}
