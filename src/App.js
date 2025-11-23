import { useState, useEffect } from 'react';
import { FileText } from 'lucide-react';

export default function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [tacticalImageFile, setTacticalImageFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [systemHealth, setSystemHealth] = useState(null);
  const [showHealth, setShowHealth] = useState(false);
  const [isTacticalMode, setIsTacticalMode] = useState(false);
  const [scenario, setScenario] = useState('Analysis request');

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
      setTacticalImageFile(null);
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
      });

      const data = await response.json();

      if (response.ok && data.success) {
        setMessages([]);
        setInputValue('');
        setUploadedFile(null);
        setTacticalImageFile(null);
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

  const handlePaste = async (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      
      if (item.type.indexOf('image') !== -1) {
        e.preventDefault();
        
        const blob = item.getAsFile();
        if (!blob) continue;

        const file = new File([blob], `pasted-img-${Date.now()}.png`, { type: blob.type });
        
        if (isTacticalMode) {
          setTacticalImageFile(file);
          setUploadedFile(file.name);
          
          const infoMsg = {
            role: 'system',
            text: `✓ Tactical image loaded: ${file.name}\n\nSend a message to run analysis.`,
            isError: false
          };
          setMessages(prev => [...prev, infoMsg]);
        } else {
          await uploadDocument(file, false);
        }
        
        break;
      }
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

  const handleTacticalAnalysis = async (file) => {
    const analysisMsg = {
      role: 'system',
      text: 'Running multi-model analysis...\nETA: 60-90s\nModels: CLIP → YOLO → LLaVA → KB → Llama',
      isError: false
    };
    setMessages(prev => [...prev, analysisMsg]);
    setIsUploading(true);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('scenario', scenario);
    
    const units = {
      infantry: 20,
      tanks: 5,
      artillery: 3,
      reconnaissance: 2
    };
    formData.append('units', JSON.stringify(units));

    try {
      const response = await fetch('http://127.0.0.1:5001/analyze_tactical', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setMessages(prev => prev.filter(msg => !msg.text.includes('Running multi-model')));

      if (response.ok && data.success) {
        let clipInfo = '';
        if (data.clip_terrain && data.clip_terrain.terrain_type) {
          clipInfo = `\nCLIP: ${data.clip_terrain.terrain_type[0].label} (${(data.clip_terrain.terrain_type[0].confidence * 100).toFixed(0)}%)\n`;
          clipInfo += `Feature: ${data.clip_terrain.tactical_features[0].label}\n`;
        }

        const strategyMessage = {
          role: 'assistant',
          text: `═══════════════════════════════════════════════════════════\n` +
                `TACTICAL ANALYSIS COMPLETE\n` +
                `═══════════════════════════════════════════════════════════\n\n` +
                `Context: ${scenario}\n` +
                `Pipeline: ${data.models_used.join(' > ')}\n` +
                `Objects Detected: ${data.yolo_detections}\n` +
                clipInfo +
                `\n${'─'.repeat(60)}\n\n` +
                `${data.strategy}\n\n` +
                `${'─'.repeat(60)}\n` +
                `Annotated Map: ${data.annotated_map}`,
          mode: 'tactical'
        };
        setMessages(prev => [...prev, strategyMessage]);
        checkHealth();
      } else {
        const errorMessage = {
          role: 'system',
          text: `Analysis failed: ${data.error || 'Unknown error'}`,
          isError: true
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      setMessages(prev => prev.filter(msg => !msg.text.includes('Running multi-model')));
      const errorMessage = {
        role: 'system',
        text: `Error: ${error.message}`,
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsUploading(false);
      setTacticalImageFile(null);
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

    results.forEach((result, idx) => {
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

    if (isTacticalMode) {
      if (files.length > 1) {
        const errorMsg = {
          role: 'system',
          text: 'Tactical mode: Upload one image at a time. Send a message to analyze.',
          isError: true
        };
        setMessages(prev => [...prev, errorMsg]);
        event.target.value = '';
        return;
      }
      
      setTacticalImageFile(files[0]);
      setUploadedFile(files[0].name);
      
      const infoMsg = {
        role: 'system',
        text: `✓ Tactical image loaded: ${files[0].name}\n\nSend a message to run analysis.`,
        isError: false
      };
      setMessages(prev => [...prev, infoMsg]);
      
      event.target.value = '';
      return;
    }

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
    if (inputValue.trim() === '' && !tacticalImageFile) return;

    // Handle tactical analysis if image is loaded
    if (isTacticalMode && tacticalImageFile) {
      // User message first
      const userMessage = { role: 'user', text: inputValue || 'Analyze this tactical map' };
      setMessages(prev => [...prev, userMessage]);

      await handleTacticalAnalysis(tacticalImageFile);
      setInputValue('');
      return;
    }

    // If in tactical mode but no image, show error
    if (isTacticalMode && !tacticalImageFile) {
      const userMessage = { role: 'user', text: inputValue };
      setMessages(prev => [...prev, userMessage]);
      setInputValue('');

      const errorMessage = {
        role: 'system',
        text: 'Tactical mode requires an image. Upload or paste an image first, then send your analysis request.',
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
      return;
    }

    const userMessage = { role: 'user', text: inputValue };
    setMessages([...messages, userMessage]);
    setInputValue('');
    setIsThinking(true);

    try {
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

  const handleKeyPress = (e) => {
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
              {isTacticalMode ? 'Tactical Vision' : 'Simulation Assistant'}
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
            <button
              onClick={() => setIsTacticalMode(!isTacticalMode)}
              className="px-3 py-1 text-xs"
              style={{
                background: isTacticalMode ? 'var(--bg-tertiary)' : 'transparent',
                border: '1px solid var(--border-color)',
                color: 'var(--text-secondary)'
              }}
            >
              Tactical Mode
            </button>

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
              {isUploading ? 'Processing' : 'Upload Docs'}
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

      {isTacticalMode && (
        <div className="border-b px-6 py-2" style={{
          borderColor: 'var(--border-color)',
          background: 'var(--bg-secondary)'
        }}>
          <div className="max-w-6xl mx-auto flex items-center gap-4">
            <span className="text-xs" style={{ color: 'var(--text-dim)' }}>Scenario:</span>
            <input
              type="text"
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              placeholder="Enter tactical scenario context"
              className="flex-1 px-3 py-1 text-xs focus:outline-none"
              style={{
                background: 'var(--bg-primary)',
                border: '1px solid var(--border-color)',
                color: 'var(--text-primary)'
              }}
            />
          </div>
        </div>
      )}

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
                {isTacticalMode ? 'Tactical Analysis' : 'Simulation Assistant'}
              </div>
              <h1 className="text-2xl mb-4" style={{
                color: 'var(--text-primary)'
              }}>
                {isTacticalMode ? 'Multi-Model Tactical Analysis' : 'Document Intelligence System'}
              </h1>
              <p className="text-xs mb-6" style={{
                color: 'var(--text-dim)'
              }}>
                {isTacticalMode
                  ? 'Upload tactical maps for CLIP, YOLO, and LLaVA analysis'
                  : 'Upload documents or ask questions'}
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
                    <div className={`text-xs leading-relaxed whitespace-pre-wrap ${
                      message.role === 'system'
                        ? 'px-3 py-2'
                        : ''
                    }`} style={message.role === 'system' ? (
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
                      {message.text}
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
              onKeyPress={handleKeyPress}
              onPaste={handlePaste}
              placeholder={isTacticalMode && tacticalImageFile ? "Send message to analyze" : isTacticalMode ? "Upload image or describe scenario" : "Ask a question or paste an image"}
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
              disabled={(inputValue.trim() === '' && !tacticalImageFile) || isThinking}
              className={`px-3 py-2 text-xs ${
                (inputValue.trim() === '' && !tacticalImageFile) || isThinking
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