/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI } from "@google/genai";
import { Mic, MicOff, Languages, Send, RefreshCw, Volume2, Copy, Check } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// --- Utilities ---
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- Types ---
interface TranslationResult {
  detectedLanguage: 'zh' | 'en';
  translation: string;
  isComplete: boolean;
}

// --- Gemini Service ---
const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });

async function* translateTextStream(text: string) {
  if (!text.trim()) return;

  try {
    const systemInstruction = `
      You are a professional real-time translator. 
      Detect if input is Chinese or English. Translate to the other.
      The input is real-time and may be incomplete. Provide a natural, fluid translation.
      If the input is a fragment, translate the fragment. If it's a full sentence, provide a polished version.
      Output format: 
      First line: "zh" or "en" (detected language)
      Following lines: The translation text.
      Do not use JSON for streaming to reduce latency. Just the language code then the text.
    `;

    const result = await genAI.models.generateContentStream({
      model: "gemini-3-flash-preview",
      contents: [{ role: 'user', parts: [{ text }] }],
      config: {
        systemInstruction,
        temperature: 0.2, // Lower temperature for faster, more consistent results
      }
    });

    let detectedLang = '';
    let fullText = '';
    
    for await (const chunk of result) {
      const chunkText = chunk.text;
      if (!chunkText) continue;
      
      if (!detectedLang) {
        const lines = chunkText.split('\n');
        detectedLang = lines[0].trim().toLowerCase().includes('zh') ? 'zh' : 'en';
        const remaining = lines.slice(1).join('\n');
        fullText += remaining;
        yield { detectedLanguage: detectedLang as 'zh' | 'en', translation: fullText };
      } else {
        fullText += chunkText;
        yield { detectedLanguage: detectedLang as 'zh' | 'en', translation: fullText };
      }
    }
  } catch (error) {
    console.error("Translation error:", error);
  }
}

// --- Main App ---
export default function App() {
  const [input, setInput] = useState('');
  const [translation, setTranslation] = useState<{ detectedLanguage: 'zh' | 'en', translation: string } | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [copied, setCopied] = useState(false);
  const [isSpeakingSource, setIsSpeakingSource] = useState(false);
  const [isSpeakingTranslation, setIsSpeakingTranslation] = useState(false);
  
  const recognitionRef = useRef<any>(null);
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Initialize Speech Recognition
  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      // Use a more flexible language setting if possible, or stick to zh-CN which often handles English too
      recognition.lang = 'zh-CN'; 

      recognition.onresult = (event: any) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          } else {
            interimTranscript += event.results[i][0].transcript;
          }
        }

        // Prioritize final transcript but show interim for real-time feel
        const currentText = finalTranscript || interimTranscript;
        if (currentText) {
          setInput(prev => {
            // If it's a final result, we might want to append or replace
            // For this app, we replace for simplicity in "real-time" mode
            return currentText;
          });
        }
      };

      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        if (event.error === 'no-speech') {
          // Don't stop on no-speech, just keep listening
          return;
        }
        setIsRecording(false);
      };

      recognition.onend = () => {
        // Auto-restart if we're still supposed to be recording
        if (isRecording) {
          try {
            recognition.start();
          } catch (e) {
            setIsRecording(false);
          }
        }
      };

      recognitionRef.current = recognition;
    }
  }, [isRecording]);

  const toggleRecording = () => {
    if (isRecording) {
      setIsRecording(false);
      recognitionRef.current?.stop();
    } else {
      setIsRecording(true);
      try {
        recognitionRef.current?.start();
      } catch (e) {
        console.error("Start error", e);
      }
    }
  };

  // Real-time translation logic with streaming
  const performTranslation = useCallback(async (text: string) => {
    if (!text.trim()) {
      setTranslation(null);
      setIsTranslating(false);
      return;
    }

    setIsTranslating(true);
    
    // Cancel previous translation if any
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    try {
      const stream = translateTextStream(text);
      for await (const result of stream) {
        setTranslation(result);
      }
    } catch (e) {
      console.error("Stream error", e);
    } finally {
      setIsTranslating(false);
    }
  }, []);

  useEffect(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    if (input) {
      // Shorter debounce for faster response
      const delay = input.length < 5 ? 200 : 400;
      debounceTimerRef.current = setTimeout(() => {
        performTranslation(input);
      }, delay);
    } else {
      setTranslation(null);
    }

    return () => {
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    };
  }, [input, performTranslation]);

  const copyToClipboard = () => {
    if (translation?.translation) {
      navigator.clipboard.writeText(translation.translation);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const speak = async (text: string, lang: 'zh' | 'en', isSource: boolean) => {
    if (!text) return;

    const setter = isSource ? setIsSpeakingSource : setIsSpeakingTranslation;
    setter(true);

    try {
      // Try Gemini TTS for high quality translation audio
      if (!isSource && text.length < 500) {
        const voiceName = lang === 'en' ? 'Puck' : 'Kore';
        const response = await genAI.models.generateContent({
          model: "gemini-2.5-flash-preview-tts",
          contents: [{ parts: [{ text }] }],
          config: {
            responseModalities: ["AUDIO" as any],
            speechConfig: {
              voiceConfig: {
                prebuiltVoiceConfig: { voiceName },
              },
            },
          },
        });

        const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
        if (base64Audio) {
          const audioUrl = `data:audio/mp3;base64,${base64Audio}`;
          if (audioRef.current) {
            audioRef.current.src = audioUrl;
            audioRef.current.onended = () => setter(false);
            audioRef.current.play();
            return;
          }
        }
      }

      // Fallback to Web Speech API
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = lang === 'zh' ? 'zh-CN' : 'en-US';
      utterance.onend = () => setter(false);
      utterance.onerror = () => setter(false);
      window.speechSynthesis.speak(utterance);
    } catch (error) {
      console.error("Speech error:", error);
      setter(false);
    }
  };

  const speakSource = () => {
    const lang = translation?.detectedLanguage === 'en' ? 'en' : 'zh';
    speak(input, lang as 'zh' | 'en', true);
  };

  const speakTranslation = () => {
    if (translation?.translation) {
      const lang = translation.detectedLanguage === 'zh' ? 'en' : 'zh';
      speak(translation.translation, lang as 'zh' | 'en', false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f8f9fa] text-[#1a1a1a] font-sans selection:bg-blue-100">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 h-16 bg-white/80 backdrop-blur-md border-b border-gray-100 z-50 flex items-center justify-between px-6">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <Languages className="text-white w-5 h-5" />
          </div>
          <h1 className="font-semibold text-lg tracking-tight">LinguaFlow</h1>
        </div>
        <div className="flex items-center gap-4 text-xs font-medium text-gray-500 uppercase tracking-widest">
          <span className={cn(translation?.detectedLanguage === 'zh' ? 'text-blue-600' : '')}>Chinese</span>
          <RefreshCw className="w-3 h-3" />
          <span className={cn(translation?.detectedLanguage === 'en' ? 'text-blue-600' : '')}>English</span>
        </div>
      </header>

      <main className="pt-24 pb-32 px-6 max-w-3xl mx-auto min-h-screen flex flex-col">
        {/* Translation Display Area */}
        <div className="flex-grow flex flex-col justify-center space-y-12 py-12">
          <AnimatePresence mode="wait">
            {!input ? (
              <motion.div
                key="empty"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="text-center space-y-4"
              >
                <p className="text-3xl font-light text-gray-400">Start typing or speak to translate</p>
                <p className="text-sm text-gray-300">Supports real-time bidirectional translation</p>
              </motion.div>
            ) : (
              <motion.div
                key="content"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-12"
              >
                {/* Source Text */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-gray-400">Source</span>
                    <div className="flex items-center gap-2">
                      {input && (
                        <button 
                          onClick={speakSource}
                          className={cn(
                            "p-2 rounded-full transition-all",
                            isSpeakingSource ? "text-blue-600 bg-blue-50" : "text-gray-400 hover:bg-gray-100 hover:text-gray-600"
                          )}
                          title="Listen Source"
                        >
                          <Volume2 className={cn("w-4 h-4", isSpeakingSource && "animate-pulse")} />
                        </button>
                      )}
                      {isTranslating && (
                        <motion.div 
                          animate={{ rotate: 360 }}
                          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                        >
                          <RefreshCw className="w-3 h-3 text-blue-500" />
                        </motion.div>
                      )}
                    </div>
                  </div>
                  <p className="text-2xl font-medium leading-relaxed text-gray-800">
                    {input}
                  </p>
                </div>

                {/* Divider */}
                <div className="h-px bg-gray-100 w-full" />

                {/* Target Text */}
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-blue-500">Translation</span>
                    {translation && (
                      <div className="flex items-center gap-2">
                        <button 
                          onClick={speakTranslation}
                          className={cn(
                            "p-2 rounded-full transition-all",
                            isSpeakingTranslation ? "text-blue-600 bg-blue-50" : "text-gray-400 hover:bg-gray-100 hover:text-blue-600"
                          )}
                          title="Listen Translation"
                        >
                          <Volume2 className={cn("w-4 h-4", isSpeakingTranslation && "animate-pulse")} />
                        </button>
                        <button 
                          onClick={copyToClipboard}
                          className="p-2 hover:bg-gray-100 rounded-full transition-colors text-gray-400 hover:text-blue-600"
                          title="Copy"
                        >
                          {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                        </button>
                      </div>
                    )}
                  </div>
                  <div className="min-h-[100px]">
                    {translation ? (
                      <motion.p 
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        className={cn(
                          "text-4xl font-light leading-tight tracking-tight",
                          translation.detectedLanguage === 'en' ? 'font-serif' : 'font-sans',
                          isTranslating && "opacity-60"
                        )}
                      >
                        {translation.translation}
                      </motion.p>
                    ) : (
                      <div className="flex gap-1 mt-4">
                        {[0, 1, 2].map((i) => (
                          <motion.div
                            key={i}
                            animate={{ opacity: [0.2, 1, 0.2] }}
                            transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.2 }}
                            className="w-2 h-2 bg-blue-200 rounded-full"
                          />
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

      {/* Input Controls */}
      <div className="fixed bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-white via-white to-transparent">
        <div className="max-w-3xl mx-auto">
          <div className="relative group">
            <div className="absolute inset-0 bg-blue-500/5 blur-2xl rounded-full group-focus-within:bg-blue-500/10 transition-all" />
            
            <div className="relative bg-white border border-gray-200 rounded-2xl shadow-xl shadow-gray-200/50 p-2 flex items-end gap-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type something..."
                className="flex-grow bg-transparent border-none focus:ring-0 resize-none py-3 px-4 max-h-40 min-h-[56px] text-lg"
                rows={1}
                onInput={(e) => {
                  const target = e.target as HTMLTextAreaElement;
                  target.style.height = 'auto';
                  target.style.height = `${target.scrollHeight}px`;
                }}
              />
              
              <div className="flex items-center gap-1 pb-1 pr-1">
                <button
                  onClick={() => setInput('')}
                  className={cn(
                    "p-3 rounded-xl transition-all",
                    input ? "text-gray-400 hover:text-gray-600 hover:bg-gray-100" : "opacity-0 pointer-events-none"
                  )}
                >
                  <RefreshCw className="w-5 h-5" />
                </button>
                
                <button
                  onClick={toggleRecording}
                  className={cn(
                    "p-3 rounded-xl transition-all relative overflow-hidden",
                    isRecording 
                      ? "bg-red-50 text-red-600" 
                      : "bg-blue-600 text-white hover:bg-blue-700 shadow-lg shadow-blue-200"
                  )}
                >
                  {isRecording ? (
                    <>
                      <MicOff className="w-6 h-6 relative z-10" />
                      <motion.div
                        animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0, 0.5] }}
                        transition={{ duration: 2, repeat: Infinity }}
                        className="absolute inset-0 bg-red-200 rounded-full"
                      />
                    </>
                  ) : (
                    <Mic className="w-6 h-6" />
                  )}
                </button>
              </div>
            </div>
          </div>
          
          <div className="mt-4 flex justify-center gap-6 text-[10px] font-bold uppercase tracking-widest text-gray-400">
            <div className="flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-green-500" />
              AI Powered
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-blue-500" />
              Real-time
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-purple-500" />
              Voice Support
            </div>
          </div>
        </div>
      </div>

      {/* Custom Styles for Textarea Scrollbar */}
      <audio ref={audioRef} className="hidden" />
      <style>{`
        textarea::-webkit-scrollbar {
          width: 4px;
        }
        textarea::-webkit-scrollbar-track {
          background: transparent;
        }
        textarea::-webkit-scrollbar-thumb {
          background: #e5e7eb;
          border-radius: 10px;
        }
        textarea::-webkit-scrollbar-thumb:hover {
          background: #d1d5db;
        }
      `}</style>
    </div>
  );
}
