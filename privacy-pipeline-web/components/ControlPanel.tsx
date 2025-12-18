'use client';

import { useState } from 'react';

export default function ControlPanel() {
  const [task, setTask] = useState('health_prediction');
  const [epochs, setEpochs] = useState(50);
  const [useDP, setUseDP] = useState(true);
  const [unlearnMethod, setUnlearnMethod] = useState('gradient_ascent');
  const [iterations, setIterations] = useState(5);

  const startTraining = async () => {
    try {
      await fetch('/api/backend/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task, epochs, use_dp: useDP }),
      });
    } catch (error) {
      console.error('Failed to start training:', error);
    }
  };

  const startUnlearning = async () => {
    try {
      await fetch('/api/backend/unlearn', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ method: unlearnMethod, iterations }),
      });
    } catch (error) {
      console.error('Failed to start unlearning:', error);
    }
  };

  const startEvaluation = async () => {
    try {
      await fetch('/api/backend/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
    } catch (error) {
      console.error('Failed to start evaluation:', error);
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
      {/* Training Card */}
      <div className="bg-black border border-gray-800 rounded-lg p-8 hover:border-gray-700 transition-all flex flex-col">
        <div className="flex justify-between items-start mb-6">
          <h3 className="text-lg font-semibold text-white">Train Model</h3>
          <span className="text-xs bg-white text-black px-3 py-1 rounded-full font-medium">
            Step 1
          </span>
        </div>

        <div className="space-y-5 flex-grow">
          <div>
            <label className="block text-sm text-gray-400 mb-2 font-medium">Task</label>
            <select
              value={task}
              onChange={(e) => setTask(e.target.value)}
              className="w-full bg-black border border-gray-700 rounded-md px-4 py-3 text-sm text-white focus:border-white focus:outline-none transition-colors"
            >
              <option value="face_recognition">Face Recognition (LFW)</option>
              <option value="health_prediction">Health Prediction (UCI)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2 font-medium">Epochs</label>
            <input
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              min="1"
              max="200"
              className="w-full bg-black border border-gray-700 rounded-md px-4 py-3 text-sm text-white focus:border-white focus:outline-none transition-colors"
            />
          </div>

          <div className="flex items-center gap-3 pt-1">
            <input
              type="checkbox"
              id="dp-checkbox"
              checked={useDP}
              onChange={(e) => setUseDP(e.target.checked)}
              className="w-4 h-4 rounded border-gray-700 bg-black text-white focus:ring-white focus:ring-offset-black"
            />
            <label htmlFor="dp-checkbox" className="text-sm text-gray-300 font-medium">
              Enable Differential Privacy
            </label>
          </div>

        </div>

        <button
          onClick={startTraining}
          className="w-full bg-white text-black py-3.5 rounded-md font-semibold flex items-center justify-center gap-2 hover:bg-gray-200 transition-colors mt-6"
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
            <polygon points="5 3 19 12 5 21 5 3" />
          </svg>
          Start Training
        </button>
      </div>

      {/* Unlearning Card */}
      <div className="bg-black border border-gray-800 rounded-lg p-8 hover:border-gray-700 transition-all flex flex-col">
        <div className="flex justify-between items-start mb-6">
          <h3 className="text-lg font-semibold text-white">Machine Unlearning</h3>
          <span className="text-xs bg-white text-black px-3 py-1 rounded-full font-medium">
            Step 2
          </span>
        </div>

        <div className="space-y-5 flex-grow">
          <div>
            <label className="block text-sm text-gray-400 mb-2 font-medium">Method</label>
            <select
              value={unlearnMethod}
              onChange={(e) => setUnlearnMethod(e.target.value)}
              className="w-full bg-black border border-gray-700 rounded-md px-4 py-3 text-sm text-white focus:border-white focus:outline-none transition-colors"
            >
              <option value="gradient_ascent">Gradient Ascent</option>
              <option value="influence_based">Influence-Based</option>
              <option value="fine_tuning">Fine-Tuning</option>
              <option value="negative_gradient">Negative Gradient</option>
            </select>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2 font-medium">Iterations</label>
            <input
              type="number"
              value={iterations}
              onChange={(e) => setIterations(parseInt(e.target.value))}
              min="1"
              max="50"
              className="w-full bg-black border border-gray-700 rounded-md px-4 py-3 text-sm text-white focus:border-white focus:outline-none transition-colors"
            />
          </div>

        </div>

        <button
          onClick={startUnlearning}
          className="w-full bg-white text-black py-3.5 rounded-md font-semibold flex items-center justify-center gap-2 hover:bg-gray-200 transition-colors mt-6"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <polyline points="1 4 1 10 7 10" />
            <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
          </svg>
          Start Unlearning
        </button>
      </div>

      {/* Privacy Evaluation Card */}
      <div className="bg-black border border-gray-800 rounded-lg p-8 hover:border-gray-700 transition-all flex flex-col">
        <div className="flex justify-between items-start mb-6">
          <h3 className="text-lg font-semibold text-white">Privacy Evaluation</h3>
          <span className="text-xs bg-white text-black px-3 py-1 rounded-full font-medium">
            Step 3
          </span>
        </div>

        <div className="space-y-5 flex-grow">
          <p className="text-sm text-gray-400 leading-relaxed font-medium">
            Evaluate privacy using Membership Inference Attack (MIA) to verify data
            protection and measure privacy guarantees.
          </p>
        </div>

        <button
          onClick={startEvaluation}
          className="w-full bg-white text-black py-3.5 rounded-md font-semibold flex items-center justify-center gap-2 hover:bg-gray-200 transition-colors mt-6"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
            <circle cx="12" cy="12" r="3" />
          </svg>
          Evaluate Privacy
        </button>
      </div>
    </div>
  );
}
