import { useState, useEffect } from 'react';
import { FileText } from 'lucide-react';

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

export default function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [systemHealth, setSystemHealth] = useState(null);
  const [showHealth, setShowHealth] = useState(false);
  const [scenario, setScenario] = useState('Tactical analysis request');

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

  const handleRestart = () => {
    if (window.confirm('Clear conversation history?')) {
      setMessages([]);
      setInputValue('');
      checkHealth();
    }
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
            <button
              onClick={() => setShowHealth(!showHealth)}
              className="flex items-center gap-2 px-3 py-1 text-xs"
              style={{
                background: 'transparent',
                border: '1px solid var(--border-color)',
                color: 'var(--text-secondary)'
              }}
            >
              <span>[{systemHealth?.status === 'healthy' ? 'ONLINE' : systemHealth?.status === 'error' ? 'ERROR' : 'CHECKING'}]</span>
            </button>
          </div>

          <div className="flex items-center gap-2">
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

            <label
              className="px-3 py-1 text-xs cursor-pointer"
              style={{
                background: 'transparent',
                border: '1px solid var(--border-color)',
                color: 'var(--text-secondary)'
              }}
            >
              {isUploading ? 'Processing' : 'Upload Files'}
              <input
                type="file"
                accept=".pdf,.jpg,.jpeg,.png,.bmp,.tiff"
                onChange={handleFileUpload}
                disabled={isUploading}
                multiple
                className="hidden"
              />
            </label>

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
                <div className="p-2" style={{
                  background: 'var(--bg-primary)',
                  border: `1px solid var(--border-color)`
                }}>
                  <div style={{ color: 'var(--text-dim)' }} className="mb-1">LLM Engine</div>
                  <div style={{
                    color: getHealthColor(systemHealth.components.ollama?.status) === 'green' ? 'var(--success)' : 'var(--error)'
                  }}>
                    {systemHealth.components.ollama?.status}
                  </div>
                </div>

                <div className="p-2" style={{
                  background: 'var(--bg-primary)',
                  border: `1px solid var(--border-color)`
                }}>
                  <div style={{ color: 'var(--text-dim)' }} className="mb-1">Embeddings</div>
                  <div style={{
                    color: getHealthColor(systemHealth.components.embeddings?.status) === 'green' ? 'var(--success)' : 'var(--error)'
                  }}>
                    {systemHealth.components.embeddings?.status}
                  </div>
                </div>

                <div className="p-2" style={{
                  background: 'var(--bg-primary)',
                  border: `1px solid var(--border-color)`
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

      <div className={`flex-1 ${messages.length > 0 ? 'overflow-y-auto' : 'overflow-hidden flex items-center justify-center'}`}>
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
                    <div className="flex-shrink-0 w-6 h-6 flex items-center justify-center text-xs" style={{
                      background: 'var(--bg-tertiary)',
                      border: '1px solid var(--border-color)',
                      color: 'var(--text-dim)'
                    }}>
                      {message.role === 'user' ? 'U' : 'A'}
                    </div>
                  )}

                  <div className={`flex-1 ${message.role === 'user' ? 'text-right' : 'text-left'}`}>
                    <div className={`text-xs leading-relaxed ${
                      message.role === 'system'
                        ? 'px-3 py-2'
                        : ''
                    } ${message.role === 'user' ? 'whitespace-pre-wrap' : ''}`} style={message.role === 'system' ? (
                      message.isError ? {
                        color: 'var(--error)',
                        background: 'var(--bg-secondary)',
                        border: '1px solid var(--border-color)'
                      } : {
                        color: 'var(--success)',
                        background: 'var(--bg-secondary)',
                        border: '1px solid var(--border-color)'
                      }
                    ) : { color: 'var(--text-primary)' }}>
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

      <div className="border-t-2 px-6 py-5" style={{
        borderColor: 'var(--border-color)',
        background: 'var(--bg-secondary)'
      }}>
        <div className="max-w-6xl mx-auto">
          <div className="relative flex items-center gap-2">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              onPaste={handlePaste}
              placeholder="Ask doctrine questions or provide coordinates (e.g., 40.7128, -74.0060)"
              rows={1}
              disabled={isThinking}
              className="flex-1 resize-none px-3 py-2 text-xs focus:outline-none disabled:opacity-50"
              style={{
                minHeight: '32px',
                maxHeight: '200px',
                background: 'var(--bg-primary)',
                border: '1px solid var(--border-color)',
                color: 'var(--text-primary)'
              }}
            />
            <button
              onClick={handleSendMessage}
              disabled={inputValue.trim() === '' || isThinking}
              className={`px-3 py-2 text-xs ${
                inputValue.trim() === '' || isThinking
                  ? 'cursor-not-allowed opacity-30'
                  : ''
              }`}
              style={{
                background: 'transparent',
                color: 'var(--text-secondary)',
                border: '1px solid var(--border-color)'
              }}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}