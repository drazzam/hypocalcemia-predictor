import React, { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Area, ComposedChart, ScatterChart, Scatter, Cell, RadialBarChart, RadialBar, PolarGrid, PolarAngleAxis, RadarChart, PolarRadiusAxis, Radar } from 'recharts';
import { Activity, TrendingUp, Zap, AlertCircle, Info, RefreshCw, Layers, Target, BarChart3, GitCompare, Sparkles, Brain, Lightbulb, Moon, Sun, Users, TrendingDown, Crosshair, Shuffle, X } from 'lucide-react';

const HypocalcemiaPredictor = () => {
  // ========== STATE ==========
  const [useSmote, setUseSmote] = useState(false);
  const [activeTab, setActiveTab] = useState('prediction');
  const [darkMode, setDarkMode] = useState(false);
  
  const [inputs, setInputs] = useState({
    calcium: 2.26,
    bmi: 28.0,
    tsh: 1.70,
    age: 41,
    magnesium: 0.79
  });

  const [showCounterfactual, setShowCounterfactual] = useState(false);
  const [targetRisk, setTargetRisk] = useState(0.05);
  const [sensitivityRange, setSensitivityRange] = useState(0.10);
  const [trajectoryDays, setTrajectoryDays] = useState(7);
  
  const componentRef = useRef(null);

  // ========== METADATA ==========
  const featureMetadata = {
    calcium: {
      name: 'Serum Calcium',
      unit: 'mmol/L',
      rank: 1,
      min: 1.5,
      max: 3.0,
      step: 0.01,
      optimal: { noSmote: [1.95, 2.41], smote: [2.16, 2.41] },
      critical: { noSmote: 1.93, smote: 2.13 },
      icon: '🩸',
      color: '#ef4444',
      description: 'Primary predictor of hypocalcemia'
    },
    bmi: {
      name: 'Body Mass Index',
      unit: 'kg/m²',
      rank: 3,
      min: 15,
      max: 50,
      step: 0.1,
      optimal: { noSmote: [20.7, 26.5], smote: [20.7, 27.8] },
      icon: '⚖️',
      color: '#8b5cf6'
    },
    tsh: {
      name: 'Preoperative TSH',
      unit: 'mIU/L',
      rank: 2,
      min: 0.1,
      max: 10,
      step: 0.01,
      optimal: { noSmote: [0.997, 1.86], smote: [0.586, 1.54] },
      icon: '🦋',
      color: '#3b82f6'
    },
    age: {
      name: 'Age at Diagnosis',
      unit: 'years',
      rank: 4,
      min: 18,
      max: 80,
      step: 1,
      optimal: { noSmote: [34, 68.2], smote: [34, 68.2] },
      icon: '👤',
      color: '#f59e0b'
    },
    magnesium: {
      name: 'Serum Magnesium',
      unit: 'mmol/L',
      rank: 5,
      min: 0.5,
      max: 1.2,
      step: 0.01,
      optimal: { noSmote: [0.672, 0.812], smote: [0.628, 0.812] },
      icon: '⚡',
      color: '#10b981'
    }
  };

  // ========== SHAP ENGINE ==========
  const calculateSHAPContributions = useCallback((values, model) => {
    const { calcium, bmi, tsh, age, magnesium } = values;
    
    if (model === 'noSmote') {
      const calciumShap = calcium < 1.93 ? 
        0.52 * Math.pow((1.93 - calcium) / 0.43, 1.25) :
        calcium > 2.41 ? 
        0.18 * Math.pow((calcium - 2.41) / 0.09, 0.85) :
        -0.08 - 0.18 * Math.cos(Math.PI * (calcium - 1.93) / 0.48);
      
      const bmiShap = bmi >= 27.4 ?
        0.12 * Math.log(1 + (bmi - 27.4) / 5) :
        -0.06 * Math.exp(-Math.pow((27.4 - bmi) / 10, 2));
      
      const tshShap = tsh < 1.0 ?
        0.10 * Math.pow((1.0 - tsh) / 0.9, 0.95) :
        tsh > 1.86 ?
        0.10 * Math.pow((tsh - 1.86) / 2.14, 0.85) :
        -0.05 * Math.exp(-Math.pow((tsh - 1.43) / 0.5, 2));
      
      const ageShap = age < 34 ?
        0.14 * Math.pow((34 - age) / 16, 1.15) :
        -0.07 * Math.exp(-Math.pow((age - 51) / 20, 2));
      
      const mgShap = magnesium < 0.659 ?
        0.11 * Math.pow((0.659 - magnesium) / 0.159, 0.95) :
        magnesium > 0.812 ?
        0.03 * (magnesium - 0.812) / 0.188 :
        -0.06 * Math.exp(-Math.pow((magnesium - 0.742) / 0.08, 2));
      
      return { calcium: calciumShap, bmi: bmiShap, tsh: tshShap, age: ageShap, magnesium: mgShap };
    } else {
      const calciumShap = calcium < 2.13 ?
        0.58 * Math.pow((2.13 - calcium) / 0.63, 1.2) :
        calcium > 2.41 ?
        0.20 * Math.pow((calcium - 2.41) / 0.09, 0.9) :
        -0.14 * Math.exp(-Math.pow((calcium - 2.27) / 0.15, 2));
      
      const bmiShap = bmi >= 27.8 && bmi <= 38.3 ?
        0.22 * Math.sin(Math.PI * (bmi - 27.8) / 10.5) :
        -0.10 * Math.exp(-Math.pow(Math.min(Math.abs(bmi - 27.8), Math.abs(bmi - 38.3)) / 8, 2));
      
      const tshShap = tsh >= 1.54 ?
        0.16 * Math.log(1 + (tsh - 1.54) / 2) :
        -0.07 * Math.exp(-Math.pow((1.54 - tsh) / 1.2, 2));
      
      const ageShap = age < 32.8 ?
        0.17 * Math.pow((32.8 - age) / 14.8, 1.1) :
        age > 50 ?
        0.04 * (age - 50) / 25 :
        -0.09 * Math.exp(-Math.pow((age - 41.4) / 10, 2));
      
      const mgShap = magnesium >= 0.672 && magnesium <= 0.769 ?
        0.14 * Math.sin(Math.PI * (magnesium - 0.672) / 0.097) :
        -0.10 * Math.exp(-Math.pow(Math.min(Math.abs(magnesium - 0.672), Math.abs(magnesium - 0.769)) / 0.15, 2));
      
      return { calcium: calciumShap, bmi: bmiShap, tsh: tshShap, age: ageShap, magnesium: mgShap };
    }
  }, []);

  // ========== RISK CALCULATION ==========
  const calculateRisk = useCallback((patientInputs, model) => {
    const baseRisk = model === 'smote' ? 0.136 : 0.046;
    const contributions = calculateSHAPContributions(patientInputs, model);
    
    const totalShap = Object.values(contributions).reduce((sum, val) => sum + val, 0);
    const logOdds = Math.log(baseRisk / (1 - baseRisk)) + totalShap * 8;
    const probability = Math.max(0.001, Math.min(0.999, 1 / (1 + Math.exp(-logOdds))));
    
    const sd = model === 'smote' ? 0.191 : 0.179;
    const epistemicUncertainty = sd / 2;
    const aleatoricUncertainty = Math.sqrt(probability * (1 - probability) / 395);
    const totalUncertainty = Math.sqrt(epistemicUncertainty ** 2 + aleatoricUncertainty ** 2);
    
    const ciLower = Math.max(0, probability - totalUncertainty);
    const ciUpper = Math.min(1, probability + totalUncertainty);
    
    let riskLevel = 'Low';
    let riskColor = 'text-green-600';
    let bgColor = darkMode ? 'bg-gradient-to-br from-green-900/40 to-green-800/40' : 'bg-gradient-to-br from-green-50 to-green-100';
    let borderColor = 'border-green-300';
    
    if (probability > 0.15) {
      riskLevel = 'High';
      riskColor = 'text-red-600';
      bgColor = darkMode ? 'bg-gradient-to-br from-red-900/40 to-red-800/40' : 'bg-gradient-to-br from-red-50 to-red-100';
      borderColor = 'border-red-300';
    } else if (probability > 0.08) {
      riskLevel = 'Moderate';
      riskColor = 'text-yellow-600';
      bgColor = darkMode ? 'bg-gradient-to-br from-yellow-900/40 to-yellow-800/40' : 'bg-gradient-to-br from-yellow-50 to-yellow-100';
      borderColor = 'border-yellow-300';
    }
    
    return {
      probability,
      ciLower,
      ciUpper,
      contributions,
      riskLevel,
      riskColor,
      bgColor,
      borderColor,
      baseRisk,
      totalShap,
      epistemicUncertainty,
      aleatoricUncertainty,
      totalUncertainty
    };
  }, [calculateSHAPContributions, darkMode]);

  const riskAnalysis = useMemo(() => {
    return calculateRisk(inputs, useSmote ? 'smote' : 'noSmote');
  }, [inputs, useSmote, calculateRisk]);

  // ========== COUNTERFACTUAL ==========
  const computeCounterfactual = useMemo(() => {
    if (!showCounterfactual) return null;
    
    const model = useSmote ? 'smote' : 'noSmote';
    let bestInputs = { ...inputs };
    const learningRate = 0.01;
    const maxIterations = 100;
    
    for (let iter = 0; iter < maxIterations; iter++) {
      const currentRisk = calculateRisk(bestInputs, model).probability;
      const riskDiff = currentRisk - targetRisk;
      
      if (Math.abs(riskDiff) < 0.005) break;
      
      const gradients = {};
      const epsilon = 0.001;
      
      for (const feature of Object.keys(featureMetadata)) {
        const perturbed = { ...bestInputs };
        perturbed[feature] += epsilon;
        const perturbedRisk = calculateRisk(perturbed, model).probability;
        gradients[feature] = (perturbedRisk - currentRisk) / epsilon;
      }
      
      for (const feature of Object.keys(featureMetadata)) {
        const meta = featureMetadata[feature];
        let update = -learningRate * riskDiff * gradients[feature];
        bestInputs[feature] = Math.max(meta.min, Math.min(meta.max, bestInputs[feature] + update));
      }
    }
    
    const changes = {};
    let totalChange = 0;
    for (const feature of Object.keys(featureMetadata)) {
      const diff = bestInputs[feature] - inputs[feature];
      changes[feature] = {
        original: inputs[feature],
        target: bestInputs[feature],
        change: diff,
        percentChange: (diff / inputs[feature]) * 100
      };
      totalChange += Math.abs(diff);
    }
    
    const achievedRisk = calculateRisk(bestInputs, model).probability;
    
    return {
      targetInputs: bestInputs,
      changes,
      totalChange,
      achievedRisk,
      feasible: totalChange < 10
    };
  }, [inputs, targetRisk, showCounterfactual, useSmote, calculateRisk, featureMetadata]);

  // ========== TRAJECTORY SIMULATOR ==========
  const riskTrajectory = useMemo(() => {
    const model = useSmote ? 'smote' : 'noSmote';
    const data = [];
    
    for (let day = 0; day <= trajectoryDays; day++) {
      const projectedInputs = {
        ...inputs,
        calcium: Math.min(2.4, inputs.calcium + (day / trajectoryDays) * 0.2)
      };
      
      const risk = calculateRisk(projectedInputs, model);
      data.push({
        day,
        risk: risk.probability * 100,
        ciLower: risk.ciLower * 100,
        ciUpper: risk.ciUpper * 100
      });
    }
    
    return data;
  }, [inputs, trajectoryDays, useSmote, calculateRisk]);

  // ========== FEATURE STABILITY ==========
  const featureStability = useMemo(() => {
    const model = useSmote ? 'smote' : 'noSmote';
    const numSamples = 100;
    const stability = {};
    
    Object.keys(featureMetadata).forEach(feature => {
      const importances = [];
      
      for (let i = 0; i < numSamples; i++) {
        const noisyInputs = { ...inputs };
        Object.keys(noisyInputs).forEach(f => {
          const meta = featureMetadata[f];
          const noise = (Math.random() - 0.5) * (meta.max - meta.min) * 0.05;
          noisyInputs[f] = Math.max(meta.min, Math.min(meta.max, noisyInputs[f] + noise));
        });
        
        const contribs = calculateSHAPContributions(noisyInputs, model);
        importances.push(Math.abs(contribs[feature]));
      }
      
      const mean = importances.reduce((a, b) => a + b, 0) / importances.length;
      const variance = importances.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / importances.length;
      const coefficientOfVariation = Math.sqrt(variance) / mean;
      
      stability[feature] = {
        mean,
        std: Math.sqrt(variance),
        cv: coefficientOfVariation,
        stable: coefficientOfVariation < 0.3
      };
    });
    
    return stability;
  }, [inputs, useSmote, calculateSHAPContributions, featureMetadata]);

  // ========== SENSITIVITY TORNADO ==========
  const sensitivityTornadoData = useMemo(() => {
    const model = useSmote ? 'smote' : 'noSmote';
    const baselineRisk = riskAnalysis.probability;
    const data = [];
    
    Object.keys(featureMetadata).forEach(feature => {
      const meta = featureMetadata[feature];
      const currentVal = inputs[feature];
      const increment = meta.step * (sensitivityRange * 100);
      
      const upInputs = { ...inputs, [feature]: Math.min(meta.max, currentVal + increment) };
      const downInputs = { ...inputs, [feature]: Math.max(meta.min, currentVal - increment) };
      
      const upRisk = calculateRisk(upInputs, model).probability;
      const downRisk = calculateRisk(downInputs, model).probability;
      
      data.push({
        feature: meta.name,
        low: (downRisk - baselineRisk) * 100,
        high: (upRisk - baselineRisk) * 100,
        range: Math.abs(upRisk - downRisk) * 100,
        color: meta.color
      });
    });
    
    return data.sort((a, b) => b.range - a.range);
  }, [inputs, riskAnalysis, sensitivityRange, useSmote, calculateRisk, featureMetadata]);

  // ========== AI INSIGHTS ==========
  const aiGeneratedInsights = useMemo(() => {
    const insights = [];
    const contribs = riskAnalysis.contributions;
    const prob = riskAnalysis.probability;
    
    insights.push({
      type: 'summary',
      icon: '🎯',
      text: `This patient presents with a ${riskAnalysis.riskLevel.toLowerCase()} risk of ${(prob * 100).toFixed(1)}% for post-thyroidectomy hypocalcemia.`
    });
    
    const sortedContribs = Object.entries(contribs)
      .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
      .slice(0, 3);
    
    sortedContribs.forEach(([feature, value]) => {
      const meta = featureMetadata[feature];
      const currentVal = inputs[feature];
      
      if (Math.abs(value) > 0.05) {
        if (value > 0) {
          insights.push({
            type: 'risk',
            icon: '⚠️',
            text: `${meta.name} (${currentVal.toFixed(feature === 'age' ? 0 : 2)} ${meta.unit}) is contributing +${(value * 800).toFixed(1)}% to risk.`
          });
        } else {
          insights.push({
            type: 'protective',
            icon: '✅',
            text: `${meta.name} (${currentVal.toFixed(feature === 'age' ? 0 : 2)} ${meta.unit}) is providing ${(value * 800).toFixed(1)}% risk reduction.`
          });
        }
      }
    });
    
    insights.push({
      type: 'uncertainty',
      icon: '📊',
      text: `Prediction confidence: ${((1 - riskAnalysis.totalUncertainty) * 100).toFixed(1)}%.`
    });
    
    return insights;
  }, [riskAnalysis, inputs, featureMetadata]);

  // ========== MODEL METRICS ==========
  const modelMetrics = useSmote ? {
    rocAuc: 0.704,
    sensitivity: 0.182,
    specificity: 0.941,
    brierScore: 0.076
  } : {
    rocAuc: 0.757,
    sensitivity: 0.182,
    specificity: 0.989,
    brierScore: 0.045
  };

  // ========== EVENT HANDLERS ==========
  const handleInputChange = useCallback((field, value) => {
    setInputs(prev => ({ ...prev, [field]: parseFloat(value) || 0 }));
  }, []);

  const handleReset = useCallback(() => {
    setInputs({ calcium: 2.26, bmi: 28.0, tsh: 1.70, age: 41, magnesium: 0.79 });
  }, []);

  const loadPreset = useCallback((preset) => {
    const presets = {
      lowRisk: { calcium: 2.30, bmi: 24.0, tsh: 1.50, age: 45, magnesium: 0.80 },
      moderateRisk: { calcium: 2.10, bmi: 32.0, tsh: 2.50, age: 35, magnesium: 0.70 },
      highRisk: { calcium: 1.85, bmi: 35.0, tsh: 3.50, age: 28, magnesium: 0.62 }
    };
    setInputs(presets[preset]);
  }, []);

  // ========== KEYBOARD SHORTCUTS ==========
  const handleKeyPress = useCallback((e) => {
    if (e.ctrlKey || e.metaKey) {
      switch(e.key) {
        case '1': setActiveTab('prediction'); e.preventDefault(); break;
        case '2': setActiveTab('shap'); e.preventDefault(); break;
        case '3': setActiveTab('advanced'); e.preventDefault(); break;
        case 'd': setDarkMode(prev => !prev); e.preventDefault(); break;
        case 'r': handleReset(); e.preventDefault(); break;
      }
    }
  }, [handleReset]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [handleKeyPress]);

  // ========== THEME ==========
  const theme = {
    bg: darkMode ? 'bg-gray-900' : 'bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100',
    card: darkMode ? 'bg-gray-800/80' : 'bg-white/80',
    text: darkMode ? 'text-gray-100' : 'text-gray-800',
    textSecondary: darkMode ? 'text-gray-400' : 'text-gray-600',
    border: darkMode ? 'border-gray-700' : 'border-gray-200',
    input: darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-300'
  };

  // ========== RENDER ==========
  return (
    <div className={`min-h-screen w-full ${theme.bg} p-6 transition-colors duration-500`} ref={componentRef}>
      <div className="max-w-[1900px] mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-4">
              <Activity className="text-indigo-600" size={48} />
              <div>
                <h1 className={`text-5xl font-black ${theme.text}`}>
                  Hypocalcemia Predictor
                </h1>
                <p className={`text-xl ${theme.textSecondary} font-medium mt-2`}>
                  Clinical ML Tool with Explainable AI
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <button
                onClick={() => setDarkMode(!darkMode)}
                className={`p-3 rounded-xl ${theme.card} border ${theme.border} hover:scale-110 transition-transform`}
                title="Toggle Dark Mode (Ctrl+D)"
              >
                {darkMode ? <Sun size={24} className="text-yellow-500" /> : <Moon size={24} className="text-indigo-600" />}
              </button>
            </div>
          </div>
          
          <p className={`text-sm ${theme.textSecondary}`}>
            Gradient Boosting | N=395 | 10-fold CV | {useSmote ? 'SMOTE Balanced Training' : 'Natural Distribution'}
          </p>
        </div>

        {/* Model Toggle */}
        <div className={`mb-6 ${theme.card} backdrop-blur-lg rounded-2xl shadow-xl p-6 border ${theme.border}`}>
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex-1">
              <h3 className={`text-lg font-bold ${theme.text} mb-2 flex items-center gap-2`}>
                <Zap className="text-yellow-500" /> Model Configuration
              </h3>
              <p className={`text-sm ${theme.textSecondary}`}>
                {useSmote 
                  ? '🔮 SMOTE: Balanced 1:1, synthetic oversampling' 
                  : '📊 Baseline: Natural 1:18 ratio'}
              </p>
            </div>
            <button
              onClick={() => setUseSmote(!useSmote)}
              className={`px-8 py-4 rounded-xl font-bold text-lg transition-all transform hover:scale-105 shadow-lg ${
                useSmote 
                  ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white' 
                  : 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white'
              }`}
            >
              {useSmote ? '🔮 SMOTE' : '📊 Baseline'}
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className={`mb-6 ${theme.card} backdrop-blur-lg rounded-2xl shadow-xl p-2 border ${theme.border}`}>
          <div className="flex gap-2">
            {[
              { id: 'prediction', label: 'Prediction', icon: Target },
              { id: 'shap', label: 'SHAP & AI', icon: Brain },
              { id: 'advanced', label: 'Advanced', icon: TrendingUp }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-semibold transition-all ${
                  activeTab === tab.id
                    ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg'
                    : `${theme.textSecondary} hover:bg-gray-100 ${darkMode ? 'hover:bg-gray-700' : ''}`
                }`}
              >
                <tab.icon size={20} />
                <span className="hidden md:inline">{tab.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        {activeTab === 'prediction' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Input Panel */}
            <div className={`${theme.card} backdrop-blur-lg rounded-2xl shadow-2xl p-8 border ${theme.border}`}>
              <div className="flex items-center justify-between mb-6">
                <h3 className={`text-2xl font-bold ${theme.text} flex items-center gap-3`}>
                  <Zap className="text-yellow-500" size={28} />
                  Patient Parameters
                </h3>
                <button
                  onClick={handleReset}
                  className={`flex items-center gap-2 px-4 py-2 ${darkMode ? 'bg-gray-700' : 'bg-gray-100'} hover:scale-105 rounded-lg transition-all`}
                >
                  <RefreshCw size={16} />
                  <span className="text-sm font-medium">Reset</span>
                </button>
              </div>

              <div className="flex gap-2 mb-6">
                <button onClick={() => loadPreset('lowRisk')} className="flex-1 px-3 py-2 bg-green-100 hover:bg-green-200 text-green-800 rounded-lg text-xs font-medium">
                  🟢 Low
                </button>
                <button onClick={() => loadPreset('moderateRisk')} className="flex-1 px-3 py-2 bg-yellow-100 hover:bg-yellow-200 text-yellow-800 rounded-lg text-xs font-medium">
                  🟡 Moderate
                </button>
                <button onClick={() => loadPreset('highRisk')} className="flex-1 px-3 py-2 bg-red-100 hover:bg-red-200 text-red-800 rounded-lg text-xs font-medium">
                  🔴 High
                </button>
              </div>
              
              <div className="space-y-6">
                {Object.entries(featureMetadata).map(([key, meta]) => (
                  <div key={key}>
                    <div className="flex items-center justify-between mb-2">
                      <label className={`text-sm font-semibold ${theme.text}`}>
                        {meta.icon} {meta.name} ({meta.unit})
                        <span className="ml-2 inline-flex items-center justify-center w-6 h-6 bg-indigo-100 text-indigo-700 rounded-full text-xs font-bold">
                          #{meta.rank}
                        </span>
                      </label>
                      <span className="text-lg font-bold text-indigo-600">
                        {inputs[key].toFixed(key === 'age' ? 0 : 2)}
                      </span>
                    </div>
                    
                    <input
                      type="range"
                      min={meta.min}
                      max={meta.max}
                      step={meta.step}
                      value={inputs[key]}
                      onChange={(e) => handleInputChange(key, e.target.value)}
                      className="w-full h-3 rounded-lg appearance-none cursor-pointer"
                      style={{
                        background: `linear-gradient(to right, 
                          ${meta.color} 0%, 
                          ${meta.color} ${((inputs[key] - meta.min) / (meta.max - meta.min)) * 100}%, 
                          ${darkMode ? '#374151' : '#e5e7eb'} ${((inputs[key] - meta.min) / (meta.max - meta.min)) * 100}%, 
                          ${darkMode ? '#374151' : '#e5e7eb'} 100%)`
                      }}
                    />
                    
                    <div className={`flex justify-between text-xs ${theme.textSecondary} mt-1`}>
                      <span>{meta.min}</span>
                      <span>{meta.max}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Risk Gauge */}
            <div className={`${riskAnalysis.bgColor} backdrop-blur-lg rounded-2xl shadow-2xl p-8 border-2 ${riskAnalysis.borderColor}`}>
              <h3 className={`text-2xl font-bold ${theme.text} mb-6 flex items-center gap-2`}>
                <Target className="text-indigo-600" size={28} />
                Risk Assessment
              </h3>
              
              <div className="text-center mb-6">
                <div className={`text-7xl font-black ${riskAnalysis.riskColor} mb-2`}>
                  {(riskAnalysis.probability * 100).toFixed(1)}%
                </div>
                <div className={`text-lg ${theme.textSecondary} font-medium mb-2`}>Predicted Probability</div>
                <div className={`text-sm ${theme.textSecondary} mb-4`}>
                  95% CI: [{(riskAnalysis.ciLower * 100).toFixed(1)}%-{(riskAnalysis.ciUpper * 100).toFixed(1)}%]
                </div>
                <div className={`inline-block px-6 py-3 rounded-full font-bold text-lg shadow-lg ${
                  riskAnalysis.riskLevel === 'High' ? 'bg-red-500 text-white' :
                  riskAnalysis.riskLevel === 'Moderate' ? 'bg-yellow-500 text-white' :
                  'bg-green-500 text-white'
                }`}>
                  {riskAnalysis.riskLevel} Risk
                </div>
              </div>

              <div className="space-y-3">
                <div className={`flex justify-between p-3 ${darkMode ? 'bg-gray-700/50' : 'bg-white/50'} rounded-lg`}>
                  <span className={`text-sm ${theme.textSecondary}`}>Base Risk:</span>
                  <span className={`font-bold ${theme.text}`}>{(riskAnalysis.baseRisk * 100).toFixed(1)}%</span>
                </div>
                <div className={`flex justify-between p-3 ${darkMode ? 'bg-blue-900/30' : 'bg-indigo-50'} rounded-lg`}>
                  <span className={`text-sm font-semibold ${darkMode ? 'text-indigo-300' : 'text-indigo-800'}`}>Model Uncertainty:</span>
                  <span className={`font-bold ${darkMode ? 'text-indigo-400' : 'text-indigo-600'}`}>
                    ±{(riskAnalysis.epistemicUncertainty * 100).toFixed(2)}%
                  </span>
                </div>
                <div className={`flex justify-between p-3 ${darkMode ? 'bg-purple-900/30' : 'bg-purple-50'} rounded-lg`}>
                  <span className={`text-sm font-semibold ${darkMode ? 'text-purple-300' : 'text-purple-800'}`}>Data Uncertainty:</span>
                  <span className={`font-bold ${darkMode ? 'text-purple-400' : 'text-purple-600'}`}>
                    ±{(riskAnalysis.aleatoricUncertainty * 100).toFixed(3)}%
                  </span>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-gray-300">
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className={`text-center p-2 ${darkMode ? 'bg-gray-700/60' : 'bg-white/60'} rounded`}>
                    <div className={`text-xs ${theme.textSecondary}`}>ROC-AUC</div>
                    <div className="font-bold text-indigo-600">{(modelMetrics.rocAuc * 100).toFixed(1)}%</div>
                  </div>
                  <div className={`text-center p-2 ${darkMode ? 'bg-gray-700/60' : 'bg-white/60'} rounded`}>
                    <div className={`text-xs ${theme.textSecondary}`}>Specificity</div>
                    <div className="font-bold text-blue-600">{(modelMetrics.specificity * 100).toFixed(1)}%</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'shap' && (
          <div className="space-y-6">
            {/* AI Insights */}
            <div className={`${theme.card} backdrop-blur-lg rounded-2xl shadow-2xl p-8 border ${theme.border}`}>
              <h3 className={`text-2xl font-bold ${theme.text} mb-4 flex items-center gap-2`}>
                <Brain className="text-purple-600" size={28} />
                AI-Generated Clinical Insights
              </h3>
              
              <div className="space-y-4">
                {aiGeneratedInsights.map((insight, idx) => (
                  <div 
                    key={idx}
                    className={`p-4 rounded-xl border-l-4 ${
                      insight.type === 'summary' ? `${darkMode ? 'bg-indigo-900/30 border-indigo-500' : 'bg-indigo-50 border-indigo-500'}` :
                      insight.type === 'risk' ? `${darkMode ? 'bg-red-900/30 border-red-500' : 'bg-red-50 border-red-500'}` :
                      insight.type === 'protective' ? `${darkMode ? 'bg-green-900/30 border-green-500' : 'bg-green-50 border-green-500'}` :
                      `${darkMode ? 'bg-purple-900/30 border-purple-500' : 'bg-purple-50 border-purple-500'}`
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <span className="text-2xl">{insight.icon}</span>
                      <p className={`text-sm ${theme.text}`}>{insight.text}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Feature Stability */}
            <div className={`${theme.card} backdrop-blur-lg rounded-2xl shadow-2xl p-8 border ${theme.border}`}>
              <h3 className={`text-2xl font-bold ${theme.text} mb-4 flex items-center gap-2`}>
                <BarChart3 className="text-teal-600" size={28} />
                Feature Stability Analysis
              </h3>
              <p className={`text-sm ${theme.textSecondary} mb-6`}>
                How stable is each feature's importance? (Lower CV = more stable)
              </p>

              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={Object.entries(featureStability).map(([feature, stats]) => ({
                  feature: featureMetadata[feature].name,
                  cv: stats.cv,
                  stable: stats.stable
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="feature" angle={-45} textAnchor="end" height={100} tick={{ fontSize: 11 }} />
                  <YAxis label={{ value: 'Coefficient of Variation', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Bar dataKey="cv" radius={[8, 8, 0, 0]}>
                    {Object.values(featureStability).map((stats, index) => (
                      <Cell key={`cell-${index}`} fill={stats.stable ? '#22c55e' : '#ef4444'} />
                    ))}
                  </Bar>
                  <ReferenceLine y={0.3} stroke="#666" strokeDasharray="5 5" label="Stability Threshold" />
                </BarChart>
              </ResponsiveContainer>

              <div className="mt-4 grid grid-cols-5 gap-2">
                {Object.entries(featureStability).map(([feature, stats]) => (
                  <div key={feature} className={`p-3 ${stats.stable ? 'bg-green-50' : 'bg-red-50'} ${darkMode && (stats.stable ? 'bg-green-900/30' : 'bg-red-900/30')} rounded-lg text-center`}>
                    <div className="text-xs font-semibold">{featureMetadata[feature].icon} {featureMetadata[feature].name.split(' ')[0]}</div>
                    <div className={`text-lg font-bold ${stats.stable ? 'text-green-600' : 'text-red-600'}`}>
                      {stats.stable ? '✓' : '⚠'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'advanced' && (
          <div className="space-y-6">
            {/* Counterfactual */}
            <div className="flex items-center gap-3 mb-4">
              <button
                onClick={() => setShowCounterfactual(!showCounterfactual)}
                className={`px-6 py-3 rounded-xl font-semibold ${
                  showCounterfactual ? 'bg-orange-600 text-white' : `${theme.card} border ${theme.border}`
                } hover:scale-105 transition-all`}
              >
                {showCounterfactual ? '✅ Counterfactual Active' : '🚀 Enable Counterfactual'}
              </button>
            </div>

            {showCounterfactual && computeCounterfactual && (
              <div className={`${theme.card} backdrop-blur-lg rounded-2xl shadow-2xl p-8 border ${theme.border}`}>
                <h3 className={`text-2xl font-bold ${theme.text} mb-4 flex items-center gap-2`}>
                  <Shuffle className="text-orange-600" size={28} />
                  Counterfactual Explorer
                </h3>

                <div className="mb-6">
                  <label className={`block text-sm font-medium ${theme.text} mb-2`}>
                    Target Risk: {(targetRisk * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="0.01"
                    max="0.30"
                    step="0.01"
                    value={targetRisk}
                    onChange={(e) => setTargetRisk(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div className={`p-4 ${computeCounterfactual.feasible ? (darkMode ? 'bg-green-900/30' : 'bg-green-50') : (darkMode ? 'bg-red-900/30' : 'bg-red-50')} rounded-lg mb-6`}>
                  <span className="text-2xl">{computeCounterfactual.feasible ? '✅' : '⚠️'}</span>
                  <span className={`ml-2 font-semibold ${theme.text}`}>
                    Achieved: {(computeCounterfactual.achievedRisk * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="space-y-3">
                  {Object.entries(computeCounterfactual.changes).map(([feature, change]) => {
                    if (Math.abs(change.change) < 0.001) return null;
                    return (
                      <div key={feature} className={`p-4 ${darkMode ? 'bg-gray-700/50' : 'bg-gray-50'} rounded-lg`}>
                        <div className="flex justify-between">
                          <span className={`font-semibold ${theme.text}`}>
                            {featureMetadata[feature].icon} {featureMetadata[feature].name}
                          </span>
                          <span className={`font-bold ${change.change > 0 ? 'text-red-600' : 'text-green-600'}`}>
                            {change.change > 0 ? '↑' : '↓'} {Math.abs(change.percentChange).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Sensitivity Tornado */}
            <div className={`${theme.card} backdrop-blur-lg rounded-2xl shadow-2xl p-8 border ${theme.border}`}>
              <h3 className={`text-2xl font-bold ${theme.text} mb-4 flex items-center gap-2`}>
                <TrendingDown className="text-teal-600" size={28} />
                Sensitivity Analysis
              </h3>

              <div className="mb-6">
                <label className={`block text-sm ${theme.text} mb-2`}>
                  Variation: ±{(sensitivityRange * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0.05"
                  max="0.25"
                  step="0.05"
                  value={sensitivityRange}
                  onChange={(e) => setSensitivityRange(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>

              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={sensitivityTornadoData} layout="vertical" margin={{ left: 120 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="feature" type="category" tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <ReferenceLine x={0} stroke="#666" strokeWidth={2} />
                  <Bar dataKey="low" stackId="a" fill="#ef4444" />
                  <Bar dataKey="high" stackId="a" fill="#22c55e" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Risk Trajectory */}
            <div className={`${theme.card} backdrop-blur-lg rounded-2xl shadow-2xl p-8 border ${theme.border}`}>
              <h3 className={`text-2xl font-bold ${theme.text} mb-4 flex items-center gap-2`}>
                <TrendingUp className="text-blue-600" size={28} />
                Risk Trajectory Simulator
              </h3>
              <p className={`text-sm ${theme.textSecondary} mb-6`}>
                Projected risk if calcium improves over time
              </p>

              <div className="mb-6">
                <label className={`block text-sm ${theme.text} mb-2`}>
                  Days: {trajectoryDays}
                </label>
                <input
                  type="range"
                  min="3"
                  max="30"
                  step="1"
                  value={trajectoryDays}
                  onChange={(e) => setTrajectoryDays(parseInt(e.target.value))}
                  className="w-full"
                />
              </div>

              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={riskTrajectory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" label={{ value: 'Days', position: 'bottom' }} />
                  <YAxis label={{ value: 'Risk %', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Area dataKey="ciUpper" fill="#8884d8" fillOpacity={0.1} stroke="none" />
                  <Area dataKey="ciLower" fill="#8884d8" fillOpacity={0.1} stroke="none" />
                  <Line type="monotone" dataKey="risk" stroke="#8b5cf6" strokeWidth={3} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        input[type="range"]::-webkit-slider-thumb {
          appearance: none;
          width: 24px;
          height: 24px;
          border-radius: 50%;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          cursor: pointer;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        
        input[type="range"]::-moz-range-thumb {
          width: 24px;
          height: 24px;
          border-radius: 50%;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          cursor: pointer;
          box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
          border: none;
        }
      `}</style>
    </div>
  );
};

export default HypocalcemiaPredictor;