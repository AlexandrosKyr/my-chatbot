import { useState, useEffect } from 'react';
import { Send, Upload, FileText, RefreshCw, AlertCircle, CheckCircle, XCircle, Trash2, Zap, Database } from 'lucide-react';

export default function Chatbot() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);
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
      const response = await fetch('http://127.0.0.1:5000/health');
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
      const response = await fetch('http://127.0.0.1:5000/delete_all', {
        method: 'POST',
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

  const handlePaste = async (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      
      if (item.type.indexOf('image') !== -1) {
        e.preventDefault();
        
        const blob = item.getAsFile();
        if (!blob) continue;

        const file = new File([blob], `img-${Date.now()}.png`, { type: blob.type });
        
        if (isTacticalMode) {
          await handleTacticalAnalysis(file);
        } else {
          await uploadDocument(file);
        }
        
        break;
      }
    }
  };

  const uploadDocument = async (file) => {
    const uploadingMsg = {
      role: 'system',
      text: 'Processing...',
      isError: false
    };
    setMessages(prev => [...prev, uploadingMsg]);
    setIsUploading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:5000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setMessages(prev => prev.filter(msg => msg.text !== 'Processing...'));

      if (response.ok && data.success) {
        setUploadedFile(file.name);
        const systemMessage = {
          role: 'system',
          text: `Indexed: ${file.name}\nChunks: ${data.details.chunks} | Size: ${data.details.file_size_kb}KB\n\nQuery ready.`,
          isError: false
        };
        setMessages(prev => [...prev, systemMessage]);
        checkHealth();
      } else {
        const errorMessage = {
          role: 'system',
          text: `Error: ${data.error}`,
          isError: true
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      setMessages(prev => prev.filter(msg => msg.text !== 'Processing...'));
      const errorMessage = {
        role: 'system',
        text: 'Connection failed.',
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsUploading(false);
    }
  };

  const handleTacticalAnalysis = async (file) => {
    const analysisMsg = {
      role: 'system',
      text: 'Running multi-model analysis...\nETA: 60-90s',
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
      const response = await fetch('http://127.0.0.1:5000/analyze_tactical_enhanced', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setMessages(prev => prev.filter(msg => !msg.text.includes('Running multi-model')));

      if (response.ok && data.success) {
        setUploadedFile(file.name);
        
        let clipInfo = '';
        if (data.clip_terrain && data.clip_terrain.terrain_type) {
          clipInfo = `\nCLIP: ${data.clip_terrain.terrain_type[0].label} (${(data.clip_terrain.terrain_type[0].confidence * 100).toFixed(0)}%)\n`;
          clipInfo += `Feature: ${data.clip_terrain.tactical_features[0].label}\n`;
        }

        const strategyMessage = {
          role: 'assistant',
          text: `ANALYSIS COMPLETE\n\n` +
                `Context: ${scenario}\n` +
                `Pipeline: ${data.models_used.join(' > ')}\n` +
                `Objects: ${data.yolo_detections}\n` +
                clipInfo +
                `\n${'─'.repeat(60)}\n\n` +
                `${data.strategy}\n\n` +
                `${'─'.repeat(60)}\n` +
                `Output: ${data.annotated_map}`,
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
    }
  };

  const handleDoctrineUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    
    setIsUploading(true);

    try {
      const response = await fetch('http://127.0.0.1:5000/upload_doctrine', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        const successMsg = {
          role: 'system',
          text: `Knowledge Base updated: ${file.name}\nConcepts indexed: ${data.chunks}`,
          isError: false
        };
        setMessages(prev => [...prev, successMsg]);
      }
    } catch (error) {
      const errorMsg = {
        role: 'system',
        text: `Upload failed: ${error.message}`,
        isError: true
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (isTacticalMode) {
      await handleTacticalAnalysis(file);
      return;
    }

    const allowedTypes = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff'];
    if (!allowedTypes.some(ext => file.name.toLowerCase().endsWith(ext))) {
      const errorMsg = {
        role: 'system',
        text: 'Invalid file type. Accepted: PDF, JPG, PNG, BMP, TIFF',
        isError: true
      };
      setMessages(prev => [...prev, errorMsg]);
      return;
    }

    await uploadDocument(file);
  };

  const handleSendMessage = async () => {
    if (inputValue.trim() === '') return;

    const userMessage = { role: 'user', text: inputValue };
    setMessages([...messages, userMessage]);
    setInputValue('');
    setIsThinking(true);

    try {
      const response = await fetch('http://127.0.0.1:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputValue }),
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
    <div className="flex flex-col h-screen bg-black text-gray-300">
      {/* Header */}
      <div className="border-b border-gray-800 bg-black px-4 py-3">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-mono text-gray-200 font-semibold">
              {isTacticalMode ? 'ENHANCED ANALYSIS' : 'SYSTEM INTERFACE'}
            </h2>
            <button
              onClick={() => setShowHealth(!showHealth)}
              className="flex items-center gap-1 text-xs"
              title="Click to see system status"
            >
              {systemHealth?.status === 'healthy' ? (
                <div className="w-3 h-3 bg-green-500 rounded-full" />
              ) : systemHealth?.status === 'error' ? (
                <div className="w-3 h-3 bg-red-500 rounded-full" />
              ) : (
                <div className="w-3 h-3 bg-yellow-500 rounded-full" />
              )}
              <span className="text-gray-400 font-mono text-xs font-semibold">STATUS</span>
            </button>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsTacticalMode(!isTacticalMode)}
              className={`px-3 py-1 text-xs font-mono font-semibold rounded transition-colors ${
                isTacticalMode 
                  ? 'bg-gray-700 text-gray-100 border border-gray-600' 
                  : 'bg-black text-gray-400 border border-gray-800 hover:border-gray-700'
              }`}
            >
              <Zap size={12} className="inline mr-1" />
              VISION
            </button>

            <label className="px-3 py-1 text-xs font-mono font-semibold rounded bg-black text-gray-400 border border-gray-800 hover:border-gray-700 cursor-pointer transition-colors" title="Upload knowledge base documents">
              <Database size={12} className="inline mr-1" />
              KB
              <input
                type="file"
                accept=".pdf,.txt,.md"
                onChange={handleDoctrineUpload}
                disabled={isUploading}
                className="hidden"
              />
            </label>

            <button
              onClick={handleDeleteAll}
              className="px-3 py-1 text-xs font-mono font-semibold rounded bg-black text-red-400 border border-gray-800 hover:border-red-700 transition-colors"
            >
              <Trash2 size={12} className="inline mr-1" />
              DELETE
            </button>

            <button
              onClick={handleRestart}
              className="px-3 py-1 text-xs font-mono font-semibold rounded bg-black text-gray-400 border border-gray-800 hover:border-gray-700 transition-colors"
            >
              <RefreshCw size={12} className="inline mr-1" />
              RESET
            </button>
            
            <label className="px-3 py-1 text-xs font-mono font-semibold rounded bg-gray-800 text-gray-100 border border-gray-700 hover:bg-gray-700 cursor-pointer transition-colors">
              <Upload size={12} className="inline mr-1" />
              {isUploading ? 'PROC...' : 'UPLOAD'}
              <input
                type="file"
                accept=".pdf,.jpg,.jpeg,.png,.bmp,.tiff"
                onChange={handleFileUpload}
                disabled={isUploading}
                className="hidden"
              />
            </label>
          </div>
        </div>
      </div>

      {/* Mode Settings */}
      {isTacticalMode && (
        <div className="border-b border-gray-800 bg-black px-4 py-2">
          <div className="max-w-3xl mx-auto flex items-center gap-4">
            <span className="text-xs font-mono text-gray-400 font-semibold">CONTEXT:</span>
            <input
              type="text"
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              placeholder="context parameters..."
              className="flex-1 px-3 py-1 text-xs font-mono bg-black border border-gray-800 rounded text-gray-300 focus:outline-none focus:border-gray-700 placeholder-gray-700"
            />
          </div>
        </div>
      )}

      {/* Health Status Panel */}
      {showHealth && systemHealth && (
        <div className="border-b border-gray-800 bg-black px-4 py-3">
          <div className="max-w-3xl mx-auto">
            <div className="flex items-start justify-between mb-3">
              <h3 className="text-sm font-mono text-gray-300 font-semibold">SYSTEM STATUS</h3>
              <button
                onClick={checkHealth}
                className="text-xs font-mono text-gray-500 hover:text-gray-300"
              >
                REFRESH
              </button>
            </div>
            
            {systemHealth.components && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono">
                <div className={`p-3 border rounded ${getHealthColor(systemHealth.components.ollama?.status) === 'green' ? 'border-green-800 bg-green-950/40' : 'border-red-800 bg-red-950/40'}`}>
                  <div className="text-gray-400 font-semibold mb-1">LLM</div>
                  <div className={`font-bold text-sm ${getHealthColor(systemHealth.components.ollama?.status) === 'green' ? 'text-green-400' : 'text-red-400'}`}>
                    {systemHealth.components.ollama?.status}
                  </div>
                </div>
                
                <div className={`p-3 border rounded ${getHealthColor(systemHealth.components.embeddings?.status) === 'green' ? 'border-green-800 bg-green-950/40' : 'border-red-800 bg-red-950/40'}`}>
                  <div className="text-gray-400 font-semibold mb-1">EMBEDDINGS</div>
                  <div className={`font-bold text-sm ${getHealthColor(systemHealth.components.embeddings?.status) === 'green' ? 'text-green-400' : 'text-red-400'}`}>
                    {systemHealth.components.embeddings?.status}
                  </div>
                </div>
                
                <div className={`p-3 border rounded ${systemHealth.components.vector_store?.status === 'ok' ? 'border-green-800 bg-green-950/40' : 'border-yellow-800 bg-yellow-950/40'}`}>
                  <div className="text-gray-400 font-semibold mb-1">VECTOR DB</div>
                  <div className={`font-bold text-sm ${systemHealth.components.vector_store?.status === 'ok' ? 'text-green-400' : 'text-yellow-400'}`}>
                    {systemHealth.components.vector_store?.status}
                  </div>
                </div>
              </div>
            )}
            
            {systemHealth.stats && (
              <div className="mt-3 pt-3 border-t border-gray-800 text-xs font-mono text-gray-400 font-semibold">
                <div className="flex gap-6">
                  <span>Queries: {systemHealth.stats.total_queries}</span>
                  <span>Documents: {systemHealth.stats.documents_processed}</span>
                  <span>Errors: {systemHealth.stats.errors}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Uploaded File Indicator */}
      {uploadedFile && (
        <div className="border-b border-gray-800 bg-black px-4 py-2">
          <div className="max-w-3xl mx-auto flex items-center gap-2 text-xs font-mono text-gray-300 font-semibold">
            <FileText size={12} className="text-gray-400" />
            <span>
              LOADED: {uploadedFile}
            </span>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      <div className={`flex-1 ${messages.length > 0 ? 'overflow-y-auto' : 'overflow-hidden flex items-center justify-center'}`}>
        <div className="max-w-3xl mx-auto px-4 w-full">
          
          {/* Welcome State */}
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center text-center px-4">
              <div className="text-xs font-mono text-gray-600 mb-2">
                {isTacticalMode ? '[ENHANCED MODE ACTIVE]' : '[SYSTEM READY]'}
              </div>
              <h1 className="text-3xl font-mono text-gray-300 mb-4 font-bold">
                {isTacticalMode ? 'AWAITING INPUT' : 'QUERY INTERFACE'}
              </h1>
              <p className="text-sm font-mono text-gray-500">
                {isTacticalMode 
                  ? 'Upload data for multi-layer analysis'
                  : 'Upload files or input query'}
              </p>
            </div>
          )}

          {/* Messages */}
          <div className="py-8 space-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`flex gap-3 max-w-full ${message.role === 'user' ? 'flex-row-reverse' : ''}`}>
                  {message.role !== 'system' && (
                    <div className={`flex-shrink-0 w-7 h-7 rounded flex items-center justify-center text-xs font-mono font-bold ${
                      message.role === 'user' 
                        ? 'bg-gray-700 text-gray-200 border border-gray-600' 
                        : message.mode === 'tactical'
                        ? 'bg-gray-700 text-gray-200 border border-gray-600'
                        : 'bg-gray-700 text-gray-200 border border-gray-600'
                    }`}>
                      {message.role === 'user' ? 'U' : 'A'}
                    </div>
                  )}
                  
                  <div className={`flex-1 ${message.role === 'user' ? 'text-right' : 'text-left'}`}>
                    <div className={`text-sm font-mono leading-relaxed whitespace-pre-wrap ${
                      message.role === 'system' 
                        ? message.isError
                          ? 'text-red-300 bg-red-950/30 border border-red-800 px-4 py-3 rounded font-semibold'
                          : 'text-green-300 bg-green-950/30 border border-green-800 px-4 py-3 rounded font-semibold'
                        : 'text-gray-200'
                    }`}>
                      {message.text}
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {isThinking && (
              <div className="flex justify-start">
                <div className="flex gap-3">
                  <div className="flex-shrink-0 w-7 h-7 rounded flex items-center justify-center text-xs font-mono font-bold bg-gray-700 text-gray-200 border border-gray-600">
                    A
                  </div>
                  <div className="text-sm font-mono text-gray-400 font-semibold">processing...</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-800 bg-black">
        <div className="max-w-3xl mx-auto px-4 py-4">
          <div className="relative flex items-end gap-2">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              onPaste={handlePaste}
              placeholder={isTacticalMode ? "input query or paste data..." : "input query..."}
              rows={1}
              disabled={isThinking}
              className="flex-1 resize-none border border-gray-800 rounded bg-black px-4 py-3 pr-12 text-sm font-mono text-gray-200 placeholder-gray-600 focus:outline-none focus:border-gray-700 disabled:opacity-50"
              style={{ minHeight: '40px', maxHeight: '200px' }}
            />
            <button
              onClick={handleSendMessage}
              disabled={inputValue.trim() === '' || isThinking}
              className={`absolute right-3 bottom-3 p-2 rounded transition-colors ${
                inputValue.trim() === '' || isThinking
                  ? 'text-gray-700 cursor-not-allowed'
                  : 'text-gray-300 hover:text-gray-100'
              }`}
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}