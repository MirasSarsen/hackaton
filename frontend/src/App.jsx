import React from "react";
import UploadForm from "./components/UploadForm";
import ResultView from "./components/ResultView";

export default function App(){
  const [result, setResult] = React.useState(null);
  return (
    <div style={{maxWidth:900, margin:"24px auto", fontFamily:"sans-serif"}}>
      <h1>AI-Translate — Demo</h1>
      <UploadForm onResult={setResult} />
      {result && <ResultView result={result} />}
    </div>
  );
}
