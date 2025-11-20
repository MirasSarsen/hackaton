import React from "react";

export default function ResultView({ result }) {
  return (
    <div style={{ marginTop: 16, padding: 12, border: "1px solid #eee" }}>
      <h3>Result</h3>

      <div><strong>Original Text:</strong></div>
      <pre style={{ whiteSpace: "pre-wrap" }}>{result.original_text}</pre>

      <div><strong>Translation:</strong></div>
      <pre style={{ whiteSpace: "pre-wrap" }}>{result.translation}</pre>

      <div><strong>Paraphrase:</strong></div>
      <pre style={{ whiteSpace: "pre-wrap" }}>{result.paraphrase}</pre>

      {result.ocr_lines && (
        <>
          <div><strong>OCR boxes:</strong></div>
          <pre>{JSON.stringify(result.ocr_lines, null, 2)}</pre>
        </>
      )}

      {result.translated_lines && (
        <>
          <div><strong>Translated chunks:</strong></div>
          <pre>{JSON.stringify(result.translated_lines, null, 2)}</pre>
        </>
      )}

      {result.translated_image_url && (
        <>
          <div><strong>Translated image:</strong></div>
          <img
            src={`http://localhost:8000/${result.translated_image_url}`}
            alt="translated"
            style={{ maxWidth: "100%", border: "1px solid #ddd", marginTop: 8 }}
          />
        </>
      )}
    </div>
  );
}
