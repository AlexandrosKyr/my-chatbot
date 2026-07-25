import { useState, useEffect, useRef } from 'react';
import { FileText, Paperclip, Send } from 'lucide-react';

// Simple markdown renderer for tactical analysis output
const renderMarkdown = (text) => {
  if (!text) return null;

  const lines = text.split('\n');
  const elements = [];
  let key = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // H2 headers (## )
    if (line.startsWith('## ')) {
      elements.push(
        <h2 key={key++} style={{
          fontSize: '14px',
          fontWeight: '600',
          marginTop: '16px',
          marginBottom: '8px',
          color: 'var(--text-primary)',
          borderBottom: '1px solid var(--border-color)',
          paddingBottom: '4px'
        }}>
          {renderInlineFormatting(line.slice(3))}
        </h2>
      );
    }
    // H3 headers (### )
    else if (line.startsWith('### ')) {
      elements.push(
        <h3 key={key++} style={{
          fontSize: '12px',
          fontWeight: '600',
          marginTop: '12px',
          marginBottom: '6px',
          color: 'var(--text-secondary)'
        }}>
          {renderInlineFormatting(line.slice(4))}
        </h3>
      );
    }
    // Horizontal rule (--- or ═══)
    else if (line.match(/^[-─═]{3,}$/)) {
      elements.push(
        <hr key={key++} style={{
          border: 'none',
          borderTop: '1px solid var(--border-color)',
          margin: '12px 0'
        }} />
      );
    }
    // Regular line
    else {
      elements.push(
        <div key={key++} style={{ minHeight: line.trim() === '' ? '8px' : 'auto' }}>
          {renderInlineFormatting(line)}
        </div>
      );
    }
  }

  return elements;
};

// Render inline formatting like **bold** and *italic*
const renderInlineFormatting = (text) => {
  if (!text) return null;

  const parts = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    // Match **bold** or *italic* (bold first to avoid partial match)
    const boldMatch = remaining.match(/\*\*(.+?)\*\*/);
    const italicMatch = remaining.match(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/);

    // Pick whichever comes first
    let match = null;
    let type = null;

    if (boldMatch && italicMatch) {
      if (boldMatch.index <= italicMatch.index) {
        match = boldMatch;
        type = 'bold';
      } else {
        match = italicMatch;
        type = 'italic';
      }
    } else if (boldMatch) {
      match = boldMatch;
      type = 'bold';
    } else if (italicMatch) {
      match = italicMatch;
      type = 'italic';
    }

    if (match) {
      const before = remaining.slice(0, match.index);
      if (before) {
        parts.push(<span key={key++}>{before}</span>);
      }
      if (type === 'bold') {
        parts.push(
          <strong key={key++} style={{ fontWeight: '600' }}>
            {match[1]}
          </strong>
        );
      } else {
        parts.push(
          <em key={key++} style={{ fontStyle: 'italic' }}>
            {match[1]}
          </em>
        );
      }
      remaining = remaining.slice(match.index + match[0].length);
    } else {
      parts.push(<span key={key++}>{remaining}</span>);
      break;
    }
  }

  return parts.length > 0 ? parts : text;
};

function Tooltip({ text, align = 'left', place = 'bottom', children }) {
  const [show, setShow] = useState(false);
  const box = {
    position: 'absolute',
    zIndex: 50,
    width: 'max-content',
    maxWidth: '300px',
    padding: '10px 13px',
    fontSize: '13.5px',
    lineHeight: 1.5,
    color: 'var(--text-primary)',
    background: 'var(--bg-tertiary)',
    border: '1px solid var(--border-color)',
    borderRadius: '10px',
    boxShadow: '0 6px 22px rgba(0, 0, 0, 0.4)',
    whiteSpace: 'normal',
    textAlign: 'left',
    pointerEvents: 'none',
    [align]: 0,
    ...(place === 'bottom' ? { top: 'calc(100% + 8px)' } : { bottom: 'calc(100% + 8px)' }),
  };
  return (
    <span
      style={{ position: 'relative', display: 'inline-flex' }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && <span style={box}>{text}</span>}
    </span>
  );
}

export default function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [systemHealth, setSystemHealth] = useState(null);
  const [showHealth, setShowHealth] = useState(false);
  const [scenario, setScenario] = useState('Tactical analysis request');

  // Composer auto-grow
  const textareaRef = useRef(null);
  const COMPOSER_MIN = 96;   // px — starting height (a few lines, like modern chat apps)
  const COMPOSER_MAX = 260;  // px — grow up to this, then scroll

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(Math.max(el.scrollHeight, COMPOSER_MIN), COMPOSER_MAX) + 'px';
  }, [inputValue]);

  // Prompt editor
  const [showPrompts, setShowPrompts] = useState(false);
  const [promptItems, setPromptItems] = useState([]);   // [{key,label,description,required,value,default}]
  const [promptDraft, setPromptDraft] = useState({});   // key -> textarea value
  const [promptErrors, setPromptErrors] = useState({}); // key -> [error strings]
  const [promptStatus, setPromptStatus] = useState('');
  const [promptSaving, setPromptSaving] = useState(false);

  useEffect(() => {
    checkHealth();
  }, []);

  useEffect(() => {
    if (messages.length > 0) {
      checkHealth();
    }
  }, [messages.length]);

  const checkHealth = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5001/health');
      const data = await response.json();
      setSystemHealth(data);
    } catch (error) {
      setSystemHealth({
        status: 'error',
        message: 'Cannot connect to backend server'
      });
    }
  };

  const handleRestart = async () => {
    if (!window.confirm('Start a fresh chat? This also removes the files you uploaded in this session. Your Knowledge Base is kept.')) {
      return;
    }
    try {
      await fetch('http://127.0.0.1:5001/delete_uploads', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
    } catch (error) {
      // Best effort — still clear the screen even if the request fails.
    }
    setMessages([]);
    setInputValue('');
    setUploadedFile(null);
    checkHealth();
  };

  const handleDeleteAll = async () => {
    if (!window.confirm('Delete all indexed data? This action cannot be reversed.')) {
      return;
    }

    try {
      const response = await fetch('http://127.0.0.1:5001/delete_all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm: true }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        setMessages([]);
        setInputValue('');
        setUploadedFile(null);
        const successMsg = {
          role: 'system',
          text: 'System reset complete. All data purged.',
          isError: false
        };
        setMessages([successMsg]);
        checkHealth();
      } else {
        const errorMsg = {
          role: 'system',
          text: `Error: ${data.error}`,
          isError: true
        };
        setMessages(prev => [...prev, errorMsg]);
      }
    } catch (error) {
      const errorMsg = {
        role: 'system',
        text: 'Connection failed.',
        isError: true
      };
      setMessages(prev => [...prev, errorMsg]);
    }
  };

  const handlePaste = () => {
    // Allow normal text paste - no special handling needed for coordinate-based analysis
  };

  const openPrompts = async () => {
    setShowPrompts(true);
    setPromptErrors({});
    setPromptStatus('Loading...');
    try {
      const response = await fetch('http://127.0.0.1:5001/prompts');
      const data = await response.json();
      const items = data.prompts || [];
      setPromptItems(items);
      const draft = {};
      items.forEach(p => { draft[p.key] = p.value; });
      setPromptDraft(draft);
      setPromptStatus('');
    } catch (error) {
      setPromptStatus('Cannot load prompts. Is the backend running?');
    }
  };

  const savePrompts = async () => {
    setPromptSaving(true);
    setPromptErrors({});
    setPromptStatus('Saving...');
    try {
      const response = await fetch('http://127.0.0.1:5001/prompts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompts: promptDraft }),
      });
      const data = await response.json();
      if (data.ok) {
        setPromptStatus('Saved. Changes apply to your next question.');
        setPromptItems(prev => prev.map(p => ({ ...p, value: promptDraft[p.key] ?? p.value })));
      } else {
        setPromptErrors(data.errors || {});
        setPromptStatus('Could not save - fix the highlighted prompts below.');
      }
    } catch (error) {
      setPromptStatus('Save failed. Is the backend running?');
    } finally {
      setPromptSaving(false);
    }
  };

  const resetPrompt = (key) => {
    const item = promptItems.find(p => p.key === key);
    if (item) {
      setPromptDraft(prev => ({ ...prev, [key]: item.default }));
    }
  };

  const uploadDocument = async (file, showCompletion = true) => {
    const uploadingMsg = {
      role: 'system',
      text: 'Processing document...',
      isError: false
    };
    
    if (showCompletion) {
      setMessages(prev => [...prev, uploadingMsg]);
    }
    
    setIsUploading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:5001/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setMessages(prev => prev.filter(msg => msg.text !== 'Processing document...'));

      if (response.ok && data.success) {
        setUploadedFile(file.name);
        
        if (showCompletion) {
          const systemMessage = {
            role: 'system',
            text: `✓ Indexed: ${file.name}\nChunks: ${data.details.chunks} | Size: ${data.details.file_size_kb}KB\n\nReady for queries.`,
            isError: false
          };
          setMessages(prev => [...prev, systemMessage]);
        }
        
        checkHealth();
        setIsUploading(false);
        return true;
      } else {
        const errorMessage = {
          role: 'system',
          text: `Error: ${data.error}`,
          isError: true
        };
        setMessages(prev => [...prev, errorMessage]);
        setIsUploading(false);
        return false;
      }
    } catch (error) {
      setMessages(prev => prev.filter(msg => msg.text !== 'Processing document...'));
      const errorMessage = {
        role: 'system',
        text: 'Connection failed.',
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
      setIsUploading(false);
      return false;
    }
  };

  const handleDoctrineUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    const uploadingMsg = {
      role: 'system',
      text: `Processing ${files.length} Knowledge Base document${files.length > 1 ? 's' : ''}...\nPlease wait...`,
      isError: false
    };
    setMessages(prev => [...prev, uploadingMsg]);

    setIsUploading(true);

    let successCount = 0;
    let failCount = 0;
    const results = [];

    for (const file of files) {
      const formData = new FormData();
      formData.append('file', file);
      
      try {
        const response = await fetch('http://127.0.0.1:5001/upload_doctrine', {
          method: 'POST',
          body: formData,
        });

        const data = await response.json();

        if (response.ok && data.success) {
          successCount++;
          results.push({
            success: true,
            filename: data.filename,
            chunks: data.chunks,
            size: data.file_size_kb
          });
        } else {
          failCount++;
          results.push({
            success: false,
            filename: file.name,
            error: data.error
          });
        }
      } catch (error) {
        failCount++;
        results.push({
          success: false,
          filename: file.name,
          error: error.message
        });
      }
    }

    setMessages(prev => prev.filter(msg => !msg.text.includes('Processing')));

    let resultText = `✓ Knowledge Base Batch Upload Complete\n\n`;
    resultText += `Success: ${successCount} | Failed: ${failCount}\n`;
    resultText += `${'─'.repeat(50)}\n\n`;

    results.forEach((result) => {
      if (result.success) {
        resultText += `✓ ${result.filename}\n  Chunks: ${result.chunks} | Size: ${result.size}KB\n\n`;
      } else {
        resultText += `✗ ${result.filename}\n  Error: ${result.error}\n\n`;
      }
    });

    resultText += `${'─'.repeat(50)}\nKB documents are permanently available for all queries.`;

    const summaryMsg = {
      role: 'system',
      text: resultText,
      isError: failCount === files.length
    };
    setMessages(prev => [...prev, summaryMsg]);
    
    setIsUploading(false);
    checkHealth();
    
    event.target.value = '';
  };

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    const allowedTypes = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff'];
    const invalidFiles = files.filter(file =>
      !allowedTypes.some(ext => file.name.toLowerCase().endsWith(ext))
    );

    if (invalidFiles.length > 0) {
      const errorMsg = {
        role: 'system',
        text: `Invalid file type(s): ${invalidFiles.map(f => f.name).join(', ')}\nAccepted: PDF, JPG, PNG, BMP, TIFF`,
        isError: true
      };
      setMessages(prev => [...prev, errorMsg]);
      event.target.value = '';
      return;
    }

    // Auto-detect: Images go to tactical analysis, PDFs to document upload
    const imageExtensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'];
    const firstFile = files[0];
    const isImage = imageExtensions.some(ext => firstFile.name.toLowerCase().endsWith(ext));

    if (isImage) {
      const infoMsg = {
        role: 'system',
        text: `ℹ️ Image files are processed via OCR for document analysis.\n\nFor tactical terrain analysis, provide coordinates in your message (e.g., "Analyze defensive positions at 48.8566, 2.3522")`,
        isError: false
      };
      setMessages(prev => [...prev, infoMsg]);

      // Process image via OCR as a document
      await uploadDocument(firstFile, true);
      event.target.value = '';
      return;
    }

    // Handle PDF document uploads
    if (files.length > 1) {
      const uploadingMsg = {
        role: 'system',
        text: `Processing ${files.length} documents...\nPlease wait...`,
        isError: false
      };
      setMessages(prev => [...prev, uploadingMsg]);

      let successCount = 0;
      let failCount = 0;

      for (const file of files) {
        const success = await uploadDocument(file, false);
        if (success) successCount++;
        else failCount++;
      }

      setMessages(prev => prev.filter(msg => !msg.text.includes('Processing')));

      const summaryMsg = {
        role: 'system',
        text: `✓ Batch Upload Complete\n\nSuccess: ${successCount} | Failed: ${failCount}\n\nReady for queries.`,
        isError: failCount === files.length
      };
      setMessages(prev => [...prev, summaryMsg]);
      checkHealth();
    } else {
      await uploadDocument(files[0], true);
    }

    event.target.value = '';
  };

  const handleSendMessage = async () => {
    if (inputValue.trim() === '') return;

    const userMessage = { role: 'user', text: inputValue };
    setMessages([...messages, userMessage]);
    setInputValue('');
    setIsThinking(true);

    // Auto-detect: Check if message contains coordinates (e.g., "40.7128, -74.0060")
    const coordPattern = /\d{1,3}\.\d{4,}/;
    const hasCoordinates = coordPattern.test(inputValue);

    try {
      if (hasCoordinates) {
        // Coordinate-based tactical analysis (long timeout for LLM inference)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 min
        const response = await fetch('http://127.0.0.1:5001/analyze_coordinates', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: userMessage.text,
            scenario: scenario
          }),
          signal: controller.signal,
        });
        clearTimeout(timeoutId);

        const data = await response.json();

        if (response.ok && data.success) {
          const coordStr = `${data.coordinates.lat.toFixed(6)}, ${data.coordinates.lon.toFixed(6)}`;
          const terrain = data.terrain_data.terrain_analysis;
          const placeName = data.terrain_data.place_name || coordStr;
          const address = data.terrain_data.address || {};
          const weather = data.terrain_data.weather || {};

          // Convert km/h to Beaufort scale
          const kmhToBeaufort = (kmh) => {
            if (kmh == null) return null;
            if (kmh < 1) return 0;
            if (kmh <= 5) return 1;
            if (kmh <= 11) return 2;
            if (kmh <= 19) return 3;
            if (kmh <= 28) return 4;
            if (kmh <= 38) return 5;
            if (kmh <= 49) return 6;
            if (kmh <= 61) return 7;
            if (kmh <= 74) return 8;
            if (kmh <= 88) return 9;
            if (kmh <= 102) return 10;
            if (kmh <= 117) return 11;
            return 12;
          };

          // Format weather summary with Beaufort scale for both avg wind and gusts
          const avgWindKmh = weather.avg_wind_speed_max_kmh;
          const gustKmh = weather.max_wind_gust_kmh;
          const avgWindStr = avgWindKmh != null ? `${avgWindKmh} km/h (Bft ${kmhToBeaufort(avgWindKmh)})` : 'N/A';
          const gustStr = gustKmh != null ? `${gustKmh} km/h (Bft ${kmhToBeaufort(gustKmh)})` : 'N/A';
          const weatherLine = weather.avg_temp_c != null
            ? `- Weather (7d): ${weather.avg_temp_c}°C avg, ${weather.total_precipitation_mm || 0}mm precip\n` +
              `- Wind (7d): avg ${avgWindStr}, gusts ${gustStr}\n`
            : '';

          const tacticalResponse = {
            role: 'assistant',
            text: `═══════════════════════════════════════════════════════════\n` +
                  `COORDINATE-BASED TACTICAL ANALYSIS\n` +
                  `═══════════════════════════════════════════════════════════\n\n` +
                  `Location: ${placeName}\n` +
                  `Coordinates: ${coordStr}\n` +
                  (address.country ? `Country: ${address.country} (${address.country_code})\n` : '') +
                  `Analysis Area: ${data.terrain_data.location.radius_km} km radius (~${(Math.PI * Math.pow(data.terrain_data.location.radius_km, 2)).toFixed(1)} km²)\n` +
                  `Scenario: ${data.scenario}\n` +
                  `Analysis Method: ${data.models_used.join(' > ')}\n\n` +
                  `Quick Terrain Assessment:\n` +
                  `- Elevation: ${data.terrain_data.elevation != null ? data.terrain_data.elevation.toFixed(1) + 'm' : 'Unknown'}\n` +
                  `- High Ground: ${terrain.high_ground ? 'YES' : 'NO'}\n` +
                  `- Cover: ${terrain.cover_availability.toUpperCase()}\n` +
                  `- Urban: ${terrain.urban_terrain ? 'YES' : 'NO'}\n` +
                  weatherLine + `\n` +
                  `${'─'.repeat(60)}\n\n` +
                  `${data.strategy}\n\n` +
                  `${'─'.repeat(60)}\n` +
                  `Data: Real terrain from OpenStreetMap + Open-Meteo Elevation API + Nominatim`,
            mode: 'coordinate_tactical'
          };
          setMessages(prev => [...prev, tacticalResponse]);
        } else {
          const errorMessage = {
            role: 'system',
            text: `Error: ${data.error}`,
            isError: true
          };
          setMessages(prev => [...prev, errorMessage]);
        }
      } else {
        // Regular chat with doctrine RAG
        const response = await fetch('http://127.0.0.1:5001/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message: userMessage.text }),
        });

        const data = await response.json();

        if (response.ok && data.success) {
          const botResponse = {
            role: 'assistant',
            text: data.response,
            mode: data.mode
          };
          setMessages(prev => [...prev, botResponse]);
        } else {
          const errorMessage = {
            role: 'system',
            text: `Error: ${data.error}`,
            isError: true
          };
          setMessages(prev => [...prev, errorMessage]);
        }
      }
    } catch (error) {
      const errorMessage = {
        role: 'system',
        text: 'Connection error. Check backend status.',
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsThinking(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const getHealthColor = (status) => {
    if (!status) return 'gray';
    if (status === 'healthy' || status === 'ok') return 'green';
    if (status === 'unhealthy' || status === 'error') return 'red';
    return 'yellow';
  };

  // ONLINE only when every engine is available: LLM, embeddings, and vector DB.
  const engines = systemHealth?.components;
  const allEnginesOk =
    engines?.ollama?.status === 'ok' &&
    engines?.embeddings?.status === 'ok' &&
    engines?.vector_store?.status === 'ok';
  const statusLabel = allEnginesOk
    ? 'ONLINE'
    : systemHealth?.status === 'error'
    ? 'OFFLINE'
    : systemHealth
    ? 'PARTIAL'
    : 'CHECKING';
  const statusColor = allEnginesOk
    ? 'var(--success)'
    : statusLabel === 'OFFLINE'
    ? 'var(--error)'
    : statusLabel === 'PARTIAL'
    ? '#fbbf24'
    : 'var(--text-dim)';

  return (
    <div className="flex flex-col h-screen" style={{ background: 'var(--bg-primary)', color: 'var(--text-primary)' }}>
      <div className="border-b-2 px-6 py-5" style={{
        borderColor: 'var(--border-color)',
        background: 'var(--bg-secondary)'
      }}>
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-6">
            <h2 className="text-sm font-medium" style={{
              color: 'var(--text-primary)'
            }}>
              Tactical Assistant
            </h2>
            <Tooltip
              align="left"
              text="Shows whether the app is ready. Green ONLINE means all three engines are working (AI, search model, and document database). Amber PARTIAL means one is still starting or missing; red OFFLINE means the server can't be reached. Click for details."
            >
              <button
                onClick={() => setShowHealth(!showHealth)}
                className="flex items-center gap-2 px-3 py-1 text-xs"
                style={{
                  background: 'transparent',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-secondary)'
                }}
              >
                <span style={{ color: statusColor, fontWeight: 600 }}>[{statusLabel}]</span>
              </button>
            </Tooltip>
          </div>

          <div className="flex items-center gap-2">
            <Tooltip
              align="right"
              text="Add reference documents (PDF, Word, or text). The assistant reads them and uses them to answer your questions. They stay saved for next time."
            >
              <label
                className="px-3 py-1 text-xs cursor-pointer"
                style={{
                  background: 'transparent',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-secondary)'
                }}
              >
                Knowledge Base
                <input
                  type="file"
                  accept=".pdf,.txt,.md,.doc,.docx"
                  onChange={handleDoctrineUpload}
                  disabled={isUploading}
                  multiple
                  className="hidden"
                />
              </label>
            </Tooltip>

            <Tooltip
              align="right"
              text="Erases everything you have added — both uploaded files and reference documents. This cannot be undone."
            >
              <button
                onClick={handleDeleteAll}
                className="px-3 py-1 text-xs"
                style={{
                  background: 'transparent',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-secondary)'
                }}
              >
                Delete All
              </button>
            </Tooltip>

            <Tooltip
              align="right"
              text="Starts a fresh chat and removes the files you uploaded here. Your Knowledge Base documents are kept."
            >
              <button
                onClick={handleRestart}
                className="px-3 py-1 text-xs"
                style={{
                  background: 'transparent',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-secondary)'
                }}
              >
                Clear Session
              </button>
            </Tooltip>

            <Tooltip
              align="right"
              text="Change the instructions that tell the assistant how to answer. You can reword them; the map, doctrine, and military information are always included automatically."
            >
              <button
                onClick={() => (showPrompts ? setShowPrompts(false) : openPrompts())}
                className="px-3 py-1 text-xs"
                style={{
                  background: 'transparent',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-secondary)'
                }}
              >
                Prompts
              </button>
            </Tooltip>
          </div>
        </div>
      </div>

      {showHealth && systemHealth && (
        <div className="border-b px-6 py-3" style={{
          borderColor: 'var(--border-color)',
          background: 'var(--bg-secondary)'
        }}>
          <div className="max-w-6xl mx-auto">
            <div className="flex items-start justify-between mb-3">
              <h3 className="text-xs" style={{ color: 'var(--text-dim)' }}>System Diagnostics</h3>
              <button
                onClick={checkHealth}
                className="px-2 py-1 text-xs"
                style={{
                  background: 'transparent',
                  color: 'var(--text-dim)',
                  border: '1px solid var(--border-color)'
                }}
              >
                Refresh
              </button>
            </div>

            {systemHealth.components && (
              <div className="grid grid-cols-3 gap-3 text-xs mb-3">
                <div className="p-3" style={{
                  background: 'var(--bg-primary)',
                  border: `1px solid var(--border-color)`,
                  borderRadius: '10px'
                }}>
                  <div style={{ color: 'var(--text-dim)' }} className="mb-1">LLM Engine</div>
                  <div style={{
                    color: getHealthColor(systemHealth.components.ollama?.status) === 'green' ? 'var(--success)' : 'var(--error)'
                  }}>
                    {systemHealth.components.ollama?.status}
                  </div>
                </div>

                <div className="p-3" style={{
                  background: 'var(--bg-primary)',
                  border: `1px solid var(--border-color)`,
                  borderRadius: '10px'
                }}>
                  <div style={{ color: 'var(--text-dim)' }} className="mb-1">Embeddings</div>
                  <div style={{
                    color: getHealthColor(systemHealth.components.embeddings?.status) === 'green' ? 'var(--success)' : 'var(--error)'
                  }}>
                    {systemHealth.components.embeddings?.status}
                  </div>
                </div>

                <div className="p-3" style={{
                  background: 'var(--bg-primary)',
                  border: `1px solid var(--border-color)`,
                  borderRadius: '10px'
                }}>
                  <div style={{ color: 'var(--text-dim)' }} className="mb-1">Vector DB</div>
                  <div style={{
                    color: systemHealth.components.vector_store?.status === 'ok' ? 'var(--success)' : 'var(--text-secondary)'
                  }}>
                    {systemHealth.components.vector_store?.status}
                  </div>
                </div>
              </div>
            )}

            {systemHealth.stats && (
              <div className="flex gap-6 text-xs pt-2" style={{ borderTop: '1px solid var(--border-color)', color: 'var(--text-dim)' }}>
                <span>queries: {systemHealth.stats.total_queries}</span>
                <span>docs: {systemHealth.stats.documents_processed}</span>
                <span>kb: {systemHealth.stats.kb_documents}</span>
                <span>errors: {systemHealth.stats.errors}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {showPrompts && (
        <div className="border-b px-6 py-3" style={{
          borderColor: 'var(--border-color)',
          background: 'var(--bg-secondary)'
        }}>
          <div className="max-w-6xl mx-auto">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs" style={{ color: 'var(--text-dim)' }}>Edit Prompts</h3>
              <div className="flex items-center gap-3">
                <span className="text-xs" style={{ color: 'var(--text-dim)' }}>{promptStatus}</span>
                <button
                  onClick={savePrompts}
                  disabled={promptSaving}
                  className="px-2 py-1 text-xs"
                  style={{ background: 'transparent', color: 'var(--text-secondary)', border: '1px solid var(--border-color)' }}
                >
                  {promptSaving ? 'Saving' : 'Save'}
                </button>
                <button
                  onClick={() => setShowPrompts(false)}
                  className="px-2 py-1 text-xs"
                  style={{ background: 'transparent', color: 'var(--text-dim)', border: '1px solid var(--border-color)' }}
                >
                  Close
                </button>
              </div>
            </div>

            <div className="space-y-4" style={{ maxHeight: '60vh', overflowY: 'auto' }}>
              {promptItems.map(p => (
                <div key={p.key} className="p-3" style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '10px' }}>
                  <div className="flex items-center justify-between mb-1">
                    <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>{p.label}</div>
                    <button
                      onClick={() => resetPrompt(p.key)}
                      className="px-2 py-1 text-xs"
                      style={{ background: 'transparent', color: 'var(--text-dim)', border: '1px solid var(--border-color)' }}
                    >
                      Reset to default
                    </button>
                  </div>
                  <div className="text-xs mb-1" style={{ color: 'var(--text-dim)' }}>{p.description}</div>
                  {p.required.length > 0 && (
                    <div className="text-xs mb-2" style={{ color: 'var(--text-secondary)' }}>
                      <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>Must keep:</span>{' '}
                      {p.required.map(r => (
                        <span
                          key={r}
                          style={{ fontWeight: 700, fontFamily: 'monospace', color: 'var(--text-primary)', marginRight: '8px' }}
                        >
                          {`{${r}}`}
                        </span>
                      ))}
                    </div>
                  )}
                  <textarea
                    value={promptDraft[p.key] ?? ''}
                    onChange={e => setPromptDraft(prev => ({ ...prev, [p.key]: e.target.value }))}
                    rows={p.key === 'ipb_analysis' ? 16 : p.key === 'followup' ? 8 : 2}
                    spellCheck={false}
                    className="w-full text-xs p-2"
                    style={{
                      background: 'var(--bg-secondary)',
                      color: 'var(--text-primary)',
                      border: `1px solid ${promptErrors[p.key] ? 'var(--error)' : 'var(--border-color)'}`,
                      fontFamily: 'monospace',
                      resize: 'vertical'
                    }}
                  />
                  {promptErrors[p.key] && (
                    <div className="text-xs mt-1" style={{ color: 'var(--error)' }}>
                      {promptErrors[p.key].join('  ·  ')}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {uploadedFile && (
        <div className="border-b px-6 py-2" style={{
          borderColor: 'var(--border-color)',
          background: 'var(--bg-secondary)'
        }}>
          <div className="max-w-6xl mx-auto flex items-center gap-2 text-xs">
            <FileText size={12} style={{ color: 'var(--text-dim)' }} />
            <span style={{ color: 'var(--text-dim)' }}>Loaded:</span>
            <span style={{ color: 'var(--text-secondary)' }}>{uploadedFile}</span>
          </div>
        </div>
      )}

      <div className={`flex-1 overflow-x-hidden ${messages.length > 0 ? 'overflow-y-auto' : 'overflow-hidden flex items-center justify-center'}`}>
        <div className="max-w-6xl mx-auto px-6 w-full">

          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center text-center px-4">
              <div className="text-xs mb-2" style={{
                color: 'var(--text-dim)'
              }}>
                Tactical Intelligence System
              </div>
              <h1 className="text-2xl mb-4" style={{
                color: 'var(--text-primary)'
              }}>
                Doctrine-Driven Analysis & Intelligence
              </h1>
              <p className="text-xs mb-6" style={{
                color: 'var(--text-dim)'
              }}>
                Provide coordinates for terrain analysis or ask doctrine questions
              </p>
              <p className="text-xs" style={{
                color: 'var(--text-dim)',
                opacity: 0.7
              }}>
                Example: "Analyze 40.7128, -74.0060 for defensive positions"
              </p>
            </div>
          )}

          <div className="py-8 space-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`flex gap-3 max-w-full ${message.role === 'user' ? 'flex-row-reverse' : ''}`}>
                  {message.role !== 'system' && (
                    <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs" style={{
                      background: 'var(--bg-tertiary)',
                      border: '1px solid var(--border-color)',
                      color: 'var(--text-secondary)'
                    }}>
                      {message.role === 'user' ? 'U' : 'A'}
                    </div>
                  )}

                  <div className={`flex-1 min-w-0 ${message.role === 'user' ? 'text-right' : 'text-left'}`} style={{ overflowWrap: 'break-word', wordBreak: 'break-word' }}>
                    <div
                      className={`text-sm leading-relaxed ${
                        message.role === 'system' || message.role === 'user' ? 'px-4 py-2' : ''
                      } ${message.role === 'user' ? 'whitespace-pre-wrap' : ''}`}
                      style={
                        message.role === 'system'
                          ? {
                              color: message.isError ? 'var(--error)' : 'var(--success)',
                              background: 'var(--bg-secondary)',
                              border: '1px solid var(--border-color)',
                              borderRadius: '12px'
                            }
                          : message.role === 'user'
                          ? {
                              color: 'var(--text-primary)',
                              background: 'var(--bg-tertiary)',
                              borderRadius: '14px',
                              display: 'inline-block',
                              textAlign: 'left'
                            }
                          : { color: 'var(--text-primary)' }
                      }
                    >
                      {message.role === 'assistant' ? renderMarkdown(message.text) : message.text}
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {isThinking && (
              <div className="flex justify-start">
                <div className="flex gap-3 items-center">
                  <div className="flex-shrink-0 w-6 h-6 flex items-center justify-center text-xs" style={{
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-color)',
                    color: 'var(--text-dim)'
                  }}>
                    A
                  </div>
                  <div className="text-xs" style={{
                    color: 'var(--text-dim)'
                  }}>Processing...</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="px-6 pt-2 pb-5" style={{
        background: 'transparent'
      }}>
        <div className="max-w-6xl mx-auto">
          <div
            className="flex flex-col px-4 pt-3 pb-2"
            style={{
              background: 'var(--bg-primary)',
              border: '1px solid var(--border-color)',
              borderRadius: '18px'
            }}
          >
            {/* Text area on top, full width */}
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              onPaste={handlePaste}
              placeholder="Ask doctrine questions or provide coordinates (e.g., 40.7128, -74.0060)"
              rows={1}
              disabled={isThinking}
              className="w-full resize-none px-1 py-1 focus:outline-none disabled:opacity-50"
              style={{
                minHeight: `${COMPOSER_MIN}px`,
                maxHeight: `${COMPOSER_MAX}px`,
                background: 'transparent',
                border: 'none',
                color: 'var(--text-primary)',
                fontSize: '15px',
                lineHeight: '1.55',
                overflowX: 'hidden',
                overflowY: 'auto'
              }}
            />

            {/* Controls row along the bottom */}
            <div className="flex items-center justify-between mt-1">
              <Tooltip
                align="left"
                place="top"
                text={isUploading
                  ? 'Reading your file, please wait...'
                  : 'Attach a PDF or image (like a map or scanned page). The assistant will read it and use it to answer your next questions.'}
              >
                <label
                  className="flex items-center justify-center cursor-pointer"
                  style={{
                    width: '38px',
                    height: '38px',
                    borderRadius: '9px',
                    color: isUploading ? 'var(--text-dim)' : 'var(--text-secondary)'
                  }}
                >
                  <Paperclip size={18} />
                  <input
                    type="file"
                    accept=".pdf,.jpg,.jpeg,.png,.bmp,.tiff"
                    onChange={handleFileUpload}
                    disabled={isUploading}
                    multiple
                    className="hidden"
                  />
                </label>
              </Tooltip>

              <div className="flex items-center gap-3">
                <span className="text-xs hidden sm:inline" style={{ color: 'var(--text-dim)' }}>
                  Enter to send · Shift+Enter for a new line
                </span>
                <Tooltip align="right" place="top" text="Send your message (or just press Enter).">
                  <button
                    onClick={handleSendMessage}
                    disabled={inputValue.trim() === '' || isThinking}
                    aria-label="Send message"
                    className={`flex items-center justify-center ${
                      inputValue.trim() === '' || isThinking
                        ? 'cursor-not-allowed opacity-40'
                        : ''
                    }`}
                    style={{
                      width: '40px',
                      height: '40px',
                      borderRadius: '10px',
                      background: 'var(--bg-tertiary)',
                      color: 'var(--text-primary)',
                      border: '1px solid var(--border-color)'
                    }}
                  >
                    <Send size={17} />
                  </button>
                </Tooltip>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}