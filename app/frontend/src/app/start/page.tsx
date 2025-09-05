'use client';

import Button from '@/components/button';
import Panel from '@/components/panel';
import Image from 'next/image';
import Link from 'next/link';
import { useState } from 'react';

interface PredictionResult {
  predicted_emotion: string;
  confidence: number;
  probabilities: Record<string, number>;
  processing_time: number;
  model_info: string;
  original_text: string;
  processed_text: string;
  error?: string;
}

export default function Start() {
  const [inputText, setInputText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!inputText.trim()) return;
    
    setIsAnalyzing(true);
    setError(null);
    setResult(null);
    
    try {
      const response = await fetch('http://localhost:8080/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: inputText
        })
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }
      
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getEmotionColor = (emotion: string) => {
    const colors = {
      anger: 'text-red-400',
      fear: 'text-purple-400',
      happiness: 'text-yellow-400',
      happy: 'text-yellow-400',
      love: 'text-pink-400',
      sadness: 'text-blue-400',
      sad: 'text-blue-400'
    };
    return colors[emotion.toLowerCase() as keyof typeof colors] || 'text-white';
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8"
         style={{ background: 'linear-gradient(135deg, #2D1B69 0%, #11052C 100%)' }}>
      <div className="max-w-4xl w-full">
        <div className="text-center mb-8">
          <Image 
            src="/logo.png"  
            alt="FeelOut Logo"  
            width={120}  
            height={120}
            className="mx-auto mb-4"
          />
          <h1 className="text-white text-3xl font-irish-grover tracking-wider mb-2 font-bold">
            <span className="text-yellow-400">Feel</span><span className="text-blue-400">Out</span> Analysis
          </h1>
          <p className="text-white/70 font-poppins">Analyze the emotional content of your text</p>
        </div>

        <Panel className="mb-8">
          <h2 className="text-yellow-400 text-2xl font-bold mb-6 font-irish-grover">Text Input</h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-white/90 font-poppins font-medium mb-2">
                Enter your text for emotion analysis:
              </label>
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="w-full h-32 text-sm px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 font-poppins resize-none focus:outline-none focus:border-yellow-400/50 focus:bg-white/15 transition-all"
                placeholder="Type your text here in Bahasa (e.g., 'Semoga keterima Lab AI' or 'Saya sedih sekali hari ini')"
                maxLength={500}
              />
              <div className="text-right text-white/50 text-xs mt-1 font-poppins">
                {inputText.length}/500 characters
              </div>
            </div>
          </div>
        </Panel>

        <div className="text-center mb-4">
          <Button
            variant="red"
            onClick={!inputText.trim() || isAnalyzing ? undefined : handleAnalyze}
            className={`${!inputText.trim() || isAnalyzing ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {isAnalyzing ? 'Analyzing...' : 'Analyze Emotion'}
          </Button>
        </div>

        {error && (
          <Panel className="mt-8 mb-8 border-red-400">
            <div className="text-red-400 font-poppins">
              <h3 className="font-semibold mb-2">Error</h3>
              <p>{error}</p>
            </div>
          </Panel>
        )}

        {result && !error && (
          <Panel className="mt-8 mb-8">
            <h2 className="text-yellow-400 text-2xl font-bold mb-6 font-irish-grover">Analysis Results</h2>
            
            <div className="space-y-6">
              <div className="text-center">
                <h3 className="text-white font-poppins font-semibold mb-2">Predicted Emotion:</h3>
                <div className={`text-4xl font-irish-grover font-bold ${getEmotionColor(result.predicted_emotion)} mb-2`}>
                  {result.predicted_emotion.toUpperCase()}
                </div>
                <div className="text-white/70 font-poppins">
                  Confidence: {(result.confidence * 100).toFixed(1)}%
                </div>
              </div>

              <div>
                <h4 className="text-white font-poppins font-semibold mb-4">Emotion Probabilities:</h4>
                <div className="space-y-2">
                  {Object.entries(result.probabilities)
                    .sort(([,a], [,b]) => b - a)
                    .map(([emotion, prob]) => (
                    <div key={emotion} className="flex items-center justify-between">
                      <span className={`font-poppins capitalize ${getEmotionColor(emotion)}`}>
                        {emotion}
                      </span>
                      <div className="flex items-center space-x-2 flex-1 ml-4">
                        <div className="flex-1 bg-white/20 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full bg-current ${getEmotionColor(emotion)}`}
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                        <span className="text-white/70 font-poppins text-sm w-12 text-right">
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex justify-between items-center text-sm font-poppins text-white/70 pt-4 border-t border-white/20">
                <span>Model: {result.model_info}</span>
                <span>Processing time: {result.processing_time}s</span>
              </div>
            </div>
          </Panel>
        )}

        <div className="text-center">
          <Link href="/">
            <Button variant="green">
              Back To Home
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}