'use client';

import Button from '@/components/button';
import Panel from '@/components/panel';
import Image from 'next/image';
import Link from 'next/link';

export default function About() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8"
         style={{ background: 'linear-gradient(135deg, #2D1B69 0%, #11052C 100%)' }}>
      <div className="max-w-4xl w-full">
        {/* Header */}
        <div className="text-center mb-8">
          <Image 
            src="/logo.png"  
            alt="FeelOut Logo"  
            width={120}  
            height={120}
            className="mx-auto mb-4"
          />
          <h1 className="text-white text-3xl font-irish-grover tracking-wider font-bold">
            About <span className="text-yellow-400">Feel</span><span className="text-blue-400">Out</span>
          </h1>
        </div>

        {/* Project Overview */}
        <Panel className="mb-8">
          <h2 className="text-yellow-400 font-irish-grover text-2xl font-bold mb-6">Project Overview</h2>
          
          <div className="font-poppins text-white/90 text-sm leading-relaxed space-y-4">
            <p>
              <strong>FeelOut</strong> is an intelligent emotion classification system designed to analyze and categorize text-based emotional expressions. This project aims to automatically identify and classify emotions from input text into one of five primary emotional categories.
            </p>
            <p>
              The system processes text input and determines which of the five core emotions the text represents: <em>anger</em>, <em>fear</em>, <em>happiness</em>, <em>love</em>, or <em>sadness</em>.
            </p>
          </div>
        </Panel>

        {/* Technical Implementation */}
        <Panel className="mb-8">
          <h3 className="text-yellow-400 font-irish-grover text-2xl font-bold mb-6">Technical Implementation</h3>
          
          <div className="font-poppins text-white/90 leading-relaxed space-y-6">
            <div>
              <h4 className="text-white font-semibold mb-2">Emotion Classification Model</h4>
              <p className="text-sm mb-4">
                This project focuses on developing an end-to-end text classification model for emotion detection, covering the complete machine learning pipeline from initial data exploration to final model validation.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h5 className="text-white/80 font-semibold mb-2 text-sm">Target Emotions</h5>
                  <ul className="list-disc list-inside space-y-1 text-sm ml-4">
                    <li><span className="text-red-400">Anger</span> - Expressions of frustration and rage</li>
                    <li><span className="text-purple-400">Fear</span> - Anxiety and apprehension</li>
                    <li><span className="text-yellow-400">Happiness</span> - Joy and positive emotions</li>
                    <li><span className="text-pink-400">Love</span> - Affection and romantic feelings</li>
                    <li><span className="text-blue-400">Sadness</span> - Sorrow and melancholy</li>
                  </ul>
                </div>
                
                <div>
                  <h5 className="text-white/80 font-semibold mb-2 text-sm">Development Pipeline</h5>
                  <ul className="list-disc list-inside space-y-1 text-sm ml-4">
                    <li>Exploratory Data Analysis (EDA)</li>
                    <li>Data preprocessing and cleaning</li>
                    <li>Feature engineering</li>
                    <li>Model training and optimization</li>
                    <li>Validation and performance evaluation</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </Panel>

        {/* Dataset Information */}
        <Panel className="mb-8">
          <h3 className="text-yellow-400 font-irish-grover text-2xl font-bold mb-6">Dataset Information</h3>
          
          <div className="font-poppins text-white/90 leading-relaxed space-y-4">
            <div>
              <h4 className="text-white font-semibold mb-2">Data Source</h4>
              <div className="bg-white/5 rounded-lg p-4 border-l-4 border-blue-400">
                <p className="text-sm mb-2">
                  <strong>Indonesian Twitter Emotion Dataset</strong>
                </p>
                <p className="text-xs text-white/70 mb-2">
                  Repository: <code className="bg-white/20 px-1 rounded">meisaputri21/Indonesian-Twitter-Emotion-Dataset</code>
                </p>
                <p className="text-xs text-white/70">
                  Purpose: Indonesian twitter dataset specifically designed for emotion classification tasks
                </p>
              </div>
            </div>
            
            <div>
              <h4 className="text-white font-semibold mb-2">Dataset Characteristics</h4>
              <ul className="list-disc list-inside space-y-1 text-sm ml-4">
                <li>Language: Indonesian (Bahasa Indonesia)</li>
                <li>Source: Twitter social media platform</li>
                <li>Task: Multi-class emotion classification</li>
                <li>Labels: 5 primary emotion categories</li>
              </ul>
            </div>
          </div>
        </Panel>

        {/* Features */}
        <Panel className="mb-8">
          <h3 className="text-yellow-400 font-irish-grover text-2xl font-bold mb-6">Key Features</h3>
          
          <div className="font-poppins text-white/90 leading-relaxed">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="text-white font-semibold mb-2">Core Functionality</h4>
                <ul className="list-disc list-inside space-y-1 text-sm ml-4">
                  <li>Text file input processing</li>
                  <li>Real-time emotion classification</li>
                  <li>Confidence scoring</li>
                </ul>
              </div>
              
              <div>
                <h4 className="text-white font-semibold mb-2">Model Performance</h4>
                <ul className="list-disc list-inside space-y-1 text-sm ml-4">
                  <li>End-to-end pipeline validation</li>
                  <li>Cross-validation testing</li>
                  <li>Performance metrics evaluation</li>
                  <li>Model interpretability analysis</li>
                </ul>
              </div>
            </div>
          </div>
        </Panel>

        {/* Author */}
        <Panel className="mb-8">
          <h3 className="text-yellow-400 font-irish-grover text-2xl font-bold mb-6">Author</h3>
          <div className='flex flex-row items-center gap-4'>
            <Image 
              src="/self.png"  
              alt="Author Profile"  
              width={100}  
              height={100}
              className="mb-4 rounded-full"
            />
            <div className='flex flex-col'>
                <p className="font-poppins text-white/90 text-md font-semibold">Adinda Putri</p>
                <p className="font-poppins text-white/90 text-sm">13523071</p>
                <p className="font-poppins text-white/70 text-sm">AI Laboratory Assistant Selection Task</p>
            </div>
          </div>
        </Panel>

        {/* Back Button */}
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