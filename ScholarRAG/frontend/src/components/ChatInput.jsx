import { useState, useRef } from "react";
import { flushSync } from "react-dom";
import { Upload, X, FileText } from "lucide-react";
import API_URL from "../api";

export default function ChatInput({ onStreamStart, onToken, onStreamEnd, onStatus, sessionId = "default", token = "", model = "claude-sonnet-4-6" }) {
  const [input, setInput]     = useState("");
  const [file, setFile]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [phase, setPhase]     = useState("");
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);

  const acceptFile = (f) => {
    if (f && f.type === "application/pdf") setFile(f);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    acceptFile(dropped);
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    const question = input;
    const fileName = file?.name || null;

    setLoading(true);
    setPhase("Searching...");
    setInput("");
    setFile(null);

    if (onStreamStart) onStreamStart(question, fileName);

    try {
      const form = new FormData();
      form.append("question", question);
      form.append("session_id", sessionId);
      form.append("model", model);
      if (file) form.append("file", file);

      const res = await fetch(`${API_URL}/query/stream`, {
        method:  "POST",
        headers: { "Authorization": `Bearer ${token}` },
        body:    form,
      });

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop();
        for (const part of parts) {
          if (!part.startsWith("data: ")) continue;
          try {
            const data = JSON.parse(part.slice(6));
            if (data.type === "status") { flushSync(() => setPhase(data.text)); if (onStatus) onStatus(data.text); }
            if (data.type === "token" && onToken) { setPhase(""); flushSync(() => onToken(data.text)); }
            if (data.type === "done"  && onStreamEnd) onStreamEnd(question, data.sources, fileName);
          } catch {}
        }
      }
    } catch (err) {
      console.error("Stream error:", err);
    } finally {
      setLoading(false);
      setPhase("");
      if (onStatus) onStatus("");
    }
  };

  return (
    <div className="w-full flex flex-col items-center gap-2">
      {phase && loading && (
        <div className="w-full max-w-3xl flex items-center gap-2 px-1">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse flex-shrink-0" />
          <span className="text-xs text-slate-400">{phase}</span>
        </div>
      )}

      {file && (
        <div className="w-full max-w-3xl flex items-center gap-2 px-3 py-2 bg-slate-800 border border-indigo-700 rounded-xl">
          <FileText size={14} className="text-indigo-400 flex-shrink-0" />
          <span className="text-sm text-slate-300 truncate flex-1">{file.name}</span>
          <button
            onClick={() => setFile(null)}
            className="text-slate-500 hover:text-slate-300 transition flex-shrink-0"
          >
            <X size={14} />
          </button>
        </div>
      )}

      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`w-full max-w-3xl bg-slate-900 border rounded-2xl p-3 flex items-center gap-3 shadow-lg transition-colors ${
          dragging ? "border-indigo-500 bg-slate-800" : "border-slate-700"
        }`}
      >
        <label className="cursor-pointer flex items-center justify-center w-9 h-9 rounded-lg bg-slate-800 hover:bg-slate-700 transition flex-shrink-0">
          <Upload size={16} className="text-slate-300" />
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            className="hidden"
            onChange={(e) => acceptFile(e.target.files[0])}
          />
        </label>

        <input
          value={dragging ? "" : input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={dragging ? "Drop PDF here..." : "Search papers, ask questions..."}
          className="flex-1 bg-transparent outline-none text-white px-2 text-[15px] placeholder:text-slate-500"
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          disabled={loading || dragging}
        />

        <button
          onClick={handleSend}
          disabled={loading}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 px-4 py-2 rounded-xl text-[15px] flex-shrink-0 transition-colors"
        >
          {loading ? (phase || "Thinking...") : "Send"}
        </button>
      </div>

      {dragging && (
        <p className="text-xs text-indigo-400">Release to attach PDF</p>
      )}

    </div>
  );
}
