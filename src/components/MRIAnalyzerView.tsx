import React, { useState, useRef } from 'react';
import {
  Upload,
  Brain,
  Sparkles,
  RotateCcw,
  Printer,
  CheckCircle2,
  AlertCircle,
  Layers,
  ArrowRight,
  ShieldCheck,
  FileText,
  Microscope,
  Gauge
} from 'lucide-react';
import { AIAnalysisResult, analyzeUploadedScan } from '../utils/aiAnalyzer';
import { HealthHeartLogo } from './HealthHeartLogo';

export const MRIAnalyzerView: React.FC = () => {
  const [equippedImage, setEquippedImage] = useState<string | null>(null);
  const [equippedFileName, setEquippedFileName] = useState<string>('');
  const [equippedFile, setEquippedFile] = useState<File | Blob | null>(null);

  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analyzingPercent, setAnalyzingPercent] = useState<number>(0);
  const [analyzingStage, setAnalyzingStage] = useState<string>('');

  const [analysisResult, setAnalysisResult] = useState<AIAnalysisResult | null>(null);
  const [showHeatmapOverlay, setShowHeatmapOverlay] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Equip uploaded file
  const handleFileUpload = (file: File) => {
    setErrorMsg(null);
    setAnalysisResult(null);

    if (!file.type.startsWith('image/')) {
      setErrorMsg('Please upload a valid image file (JPEG, PNG, WebP).');
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      setEquippedFile(file);
      setEquippedFileName(file.name);
      setEquippedImage(reader.result as string);
    };
    reader.onerror = () => {
      setErrorMsg('Failed to read image file.');
    };
    reader.readAsDataURL(file);
  };

  // Run AI analysis with health logo UI and 1% to 100% visualization
  const handleStartAnalysis = async () => {
    if (!equippedFile || !equippedImage) return;

    setErrorMsg(null);
    setIsAnalyzing(true);
    setAnalyzingPercent(1);
    setAnalyzingStage('Equipping image & initializing AI vision model...');

    try {
      const result = await analyzeUploadedScan(
        equippedFile,
        equippedFileName,
        equippedImage,
        (stage, pct) => {
          setAnalyzingStage(stage);
          setAnalyzingPercent(pct);
        }
      );

      setAnalysisResult(result);
    } catch (err: any) {
      console.error(err);
      setErrorMsg('AI Analysis encountered an issue: ' + (err.message || 'Please try another scan.'));
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Reset to analyze another image
  const handleReset = () => {
    setEquippedImage(null);
    setEquippedFileName('');
    setEquippedFile(null);
    setAnalysisResult(null);
    setIsAnalyzing(false);
    setErrorMsg(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Top Header */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-slate-900 text-white text-xs font-medium shadow-xs">
          <Sparkles className="w-3.5 h-3.5 text-teal-400" />
          <span>Neuro Clinical AI Vision</span>
        </div>
        <h1 className="text-2xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
          Brain MRI AI Health Analyzer
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 max-w-lg mx-auto">
          Upload any brain MRI or medical scan. The multimodal AI performs high-resolution structural inspection, tissue density mapping, and in-depth diagnostic evaluation.
        </p>
      </div>

      {/* Error notification */}
      {errorMsg && (
        <div className="p-3.5 bg-red-50 border border-red-200 rounded-xl text-xs text-red-800 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-red-600 shrink-0" />
            <span>{errorMsg}</span>
          </div>
          <button onClick={() => setErrorMsg(null)} className="underline font-semibold ml-2 cursor-pointer">
            Dismiss
          </button>
        </div>
      )}

      {/* Main Scanner Container */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        {/* STEP 1: Upload Prompt if no image equipped */}
        {!equippedImage && (
          <div className="p-8 sm:p-14 text-center space-y-6">
            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-slate-300 hover:border-teal-500 bg-slate-50/70 hover:bg-teal-50/30 rounded-2xl p-10 sm:p-16 text-center cursor-pointer transition-all duration-200 group"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFileUpload(e.target.files[0]);
                  }
                }}
              />
              <div className="w-16 h-16 rounded-2xl bg-white border border-slate-200 shadow-xs text-teal-600 mx-auto flex items-center justify-center mb-4 group-hover:scale-105 transition-transform">
                <Upload className="w-8 h-8 text-teal-600" />
              </div>
              <h3 className="text-base font-bold text-slate-900">
                Upload Brain MRI Scan to Equip
              </h3>
              <p className="text-xs text-slate-500 mt-1.5 max-w-sm mx-auto">
                Drag and drop your scan file here or browse from your device (JPG, PNG, WebP)
              </p>
              <div className="mt-5 inline-flex items-center space-x-2 bg-slate-900 text-white text-xs font-semibold px-4 py-2 rounded-xl group-hover:bg-slate-800 transition-colors">
                <span>Select Scan File</span>
                <ArrowRight className="w-3.5 h-3.5 text-teal-400" />
              </div>
            </div>
          </div>
        )}

        {/* STEP 2: Equipped Image Viewport (Idle, Analyzing, or Completed) */}
        {equippedImage && (
          <div className="p-6 sm:p-8 space-y-6">
            {/* Header info bar */}
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 text-xs">
              <div className="flex items-center space-x-2">
                <span className="w-2.5 h-2.5 rounded-full bg-teal-500 animate-pulse" />
                <span className="font-semibold text-slate-900">Equipped Scan:</span>
                <span className="text-slate-500 font-mono truncate max-w-[200px] sm:max-w-xs">
                  {equippedFileName}
                </span>
              </div>

              {!isAnalyzing && (
                <button
                  onClick={handleReset}
                  className="text-slate-600 hover:text-slate-900 font-medium flex items-center space-x-1 cursor-pointer bg-slate-100 hover:bg-slate-200 px-2.5 py-1 rounded-lg transition-colors"
                >
                  <RotateCcw className="w-3 h-3" />
                  <span>Choose Another Image</span>
                </button>
              )}
            </div>

            {/* Central Equipped Image Scanner Frame */}
            <div className="flex flex-col items-center justify-center">
              <div className="relative w-72 h-72 sm:w-80 sm:h-80 rounded-2xl overflow-hidden bg-black border-2 border-slate-900 shadow-2xl flex items-center justify-center">
                {/* Equipped MRI Scan */}
                <img
                  src={equippedImage}
                  alt="Equipped Brain MRI"
                  className="w-full h-full object-contain"
                />

                {/* Heatmap Overlay */}
                {analysisResult?.gradcamOverlayUrl && showHeatmapOverlay && !isAnalyzing && (
                  <img
                    src={analysisResult.gradcamOverlayUrl}
                    alt="AI Visual Focus Heatmap"
                    className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-300"
                    style={{ opacity: 0.8 }}
                  />
                )}
              </div>

              {/* Heatmap Focus Toggle */}
              {analysisResult && !isAnalyzing && (
                <div className="mt-3 flex items-center space-x-2">
                  <button
                    onClick={() => setShowHeatmapOverlay(!showHeatmapOverlay)}
                    className={`flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                      showHeatmapOverlay
                        ? 'bg-teal-100 text-teal-900 border border-teal-300 shadow-2xs'
                        : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                    }`}
                  >
                    <Layers className="w-3.5 h-3.5" />
                    <span>{showHeatmapOverlay ? 'AI Focus Heatmap: ON' : 'Show AI Focus Heatmap'}</span>
                  </button>
                </div>
              )}
            </div>

            {/* ACTION: Ready to Analyze */}
            {!isAnalyzing && !analysisResult && (
              <div className="text-center pt-2">
                <button
                  onClick={handleStartAnalysis}
                  className="bg-slate-900 hover:bg-slate-800 text-white font-bold text-sm px-8 py-3.5 rounded-xl shadow-lg hover:shadow-xl transition-all flex items-center justify-center space-x-2 mx-auto cursor-pointer group"
                >
                  <Sparkles className="w-4 h-4 text-teal-400 group-hover:rotate-12 transition-transform" />
                  <span>Start AI Medical Analysis</span>
                  <ArrowRight className="w-4 h-4 text-slate-400 group-hover:translate-x-1 transition-transform" />
                </button>
                <p className="text-xs text-slate-400 mt-2">
                  Activates multimodal neural vision to evaluate ventricles, tissues, and focal pathology.
                </p>
              </div>
            )}

            {/* ANALYZING STATE: Clean Health Logo UI with 1% ... 100% Progress */}
            {isAnalyzing && (
              <div className="max-w-md mx-auto space-y-5 pt-3 pb-2 text-center">
                {/* Health Heart Logo */}
                <div className="flex justify-center items-center py-2">
                  <div className="relative p-3 bg-white rounded-full border border-slate-100 shadow-sm">
                    <HealthHeartLogo size={140} isSpinning={true} />
                  </div>
                </div>

                {/* Percentage & Scanning Progress Bar */}
                <div className="space-y-2.5">
                  <div className="flex items-baseline justify-center space-x-1">
                    <span className="text-5xl sm:text-6xl font-black text-slate-900 font-mono tracking-tight">
                      {analyzingPercent}
                    </span>
                    <span className="text-3xl font-bold text-teal-600 font-mono">%</span>
                  </div>

                  <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden p-0.5 border border-slate-200">
                    <div
                      className="bg-slate-900 h-full rounded-full transition-all duration-200 ease-out"
                      style={{ width: `${analyzingPercent}%` }}
                    />
                  </div>

                  <p className="text-xs font-mono text-slate-600 animate-pulse pt-0.5">
                    {analyzingStage}
                  </p>
                </div>
              </div>
            )}

            {/* PREDICTION RESULT: Expanded In-Depth AI Analysis Content */}
            {analysisResult && !isAnalyzing && (
              <div className="space-y-6 pt-4 border-t border-slate-100">
                {/* 1. Primary Impression Banner */}
                <div className="bg-slate-50 rounded-2xl p-5 sm:p-6 border border-slate-200 flex flex-col sm:flex-row items-center justify-between gap-4">
                  <div className="space-y-1.5 text-center sm:text-left">
                    <div className="flex items-center justify-center sm:justify-start space-x-2">
                      <span className="text-[10px] font-bold uppercase tracking-wider bg-teal-100 text-teal-800 px-2.5 py-0.5 rounded-md">
                        {analysisResult.category}
                      </span>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-md ${
                        analysisResult.severity === 'Normal'
                          ? 'bg-emerald-100 text-emerald-800'
                          : 'bg-amber-100 text-amber-800'
                      }`}>
                        {analysisResult.severity} Severity
                      </span>
                    </div>

                    <h2 className="text-2xl sm:text-3xl font-black text-slate-900 tracking-tight flex items-center justify-center sm:justify-start space-x-2">
                      <CheckCircle2 className="w-6 h-6 text-teal-600 shrink-0" />
                      <span>{analysisResult.finding}</span>
                    </h2>
                    <p className="text-xs text-slate-500 font-mono">
                      Evaluation completed via {analysisResult.model_name}
                    </p>
                  </div>

                  <div className="text-center sm:text-right bg-white p-4 rounded-xl border border-slate-200 shadow-2xs shrink-0 min-w-[140px]">
                    <span className="text-[10px] uppercase font-semibold text-slate-400 block">
                      Model Confidence
                    </span>
                    <span className="text-3xl sm:text-4xl font-black text-slate-900 font-mono block">
                      {analysisResult.display_confidence}
                    </span>
                    <span className="text-[10px] text-teal-700 font-medium font-mono">
                      Latency: {analysisResult.inference_time_ms} ms
                    </span>
                  </div>
                </div>

                {/* 2. Quantitative Bio-metrics & Measurements */}
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Gauge className="w-4 h-4 text-teal-600" />
                    <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                      Quantitative Bio-metrics & Spatial Alignment
                    </h3>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                    <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 space-y-1">
                      <span className="text-[10px] font-semibold text-slate-400 uppercase block">Midline Shift</span>
                      <span className="text-xs font-bold text-slate-900 block">{analysisResult.biomarkers.midlineShift}</span>
                      <span className="text-[10px] text-teal-700 font-medium">Symmetry check</span>
                    </div>

                    <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 space-y-1">
                      <span className="text-[10px] font-semibold text-slate-400 uppercase block">Tissue Homogeneity</span>
                      <span className="text-xs font-bold text-slate-900 block">{analysisResult.biomarkers.tissueHomogeneity}</span>
                      <span className="text-[10px] text-teal-700 font-medium">Voxel density balance</span>
                    </div>

                    <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 space-y-1">
                      <span className="text-[10px] font-semibold text-slate-400 uppercase block">Ventricular Caliber</span>
                      <span className="text-xs font-bold text-slate-900 block">{analysisResult.biomarkers.ventricleIndex}</span>
                      <span className="text-[10px] text-teal-700 font-medium">CSF space ratio</span>
                    </div>

                    <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200 space-y-1">
                      <span className="text-[10px] font-semibold text-slate-400 uppercase block">Signal Intensity</span>
                      <span className="text-xs font-bold text-slate-900 block">{analysisResult.biomarkers.signalIntensity}</span>
                      <span className="text-[10px] text-teal-700 font-medium">T2/FLAIR profile</span>
                    </div>
                  </div>
                </div>

                {/* 3. Comprehensive Anatomical Compartment Breakdown (Expanded Content) */}
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <Microscope className="w-4 h-4 text-teal-600" />
                    <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                      Anatomical Compartment Breakdown
                    </h3>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                    {/* Ventricular System */}
                    <div className="p-4 bg-white rounded-xl border border-slate-200 space-y-1.5 shadow-2xs">
                      <div className="flex items-center space-x-2 text-slate-900 font-bold">
                        <span className="w-2 h-2 rounded-full bg-teal-500" />
                        <span>Ventricular System & CSF Pathways</span>
                      </div>
                      <p className="text-slate-600 text-xs leading-relaxed">
                        {analysisResult.ventriclesCSF}
                      </p>
                    </div>

                    {/* Hemispheric Symmetry */}
                    <div className="p-4 bg-white rounded-xl border border-slate-200 space-y-1.5 shadow-2xs">
                      <div className="flex items-center space-x-2 text-slate-900 font-bold">
                        <span className="w-2 h-2 rounded-full bg-teal-500" />
                        <span>Hemispheric Symmetry & Midline Falx</span>
                      </div>
                      <p className="text-slate-600 text-xs leading-relaxed">
                        {analysisResult.hemisphericSymmetry}
                      </p>
                    </div>

                    {/* Parenchyma Density */}
                    <div className="p-4 bg-white rounded-xl border border-slate-200 space-y-1.5 shadow-2xs">
                      <div className="flex items-center space-x-2 text-slate-900 font-bold">
                        <span className="w-2 h-2 rounded-full bg-teal-500" />
                        <span>Gray-White Matter Parenchyma</span>
                      </div>
                      <p className="text-slate-600 text-xs leading-relaxed">
                        {analysisResult.parenchymaDensity}
                      </p>
                    </div>

                    {/* Focal Pathology & Edema */}
                    <div className="p-4 bg-white rounded-xl border border-slate-200 space-y-1.5 shadow-2xs">
                      <div className="flex items-center space-x-2 text-slate-900 font-bold">
                        <span className="w-2 h-2 rounded-full bg-teal-500" />
                        <span>Focal Pathology & Mass Effect</span>
                      </div>
                      <p className="text-slate-600 text-xs leading-relaxed">
                        {analysisResult.focalFindings}
                      </p>
                    </div>
                  </div>
                </div>

                {/* 4. Detailed AI Radiological Synthesis */}
                <div className="bg-slate-900 text-white rounded-2xl p-6 space-y-4 shadow-lg">
                  <div className="flex items-center space-x-2 pb-2 border-b border-slate-800">
                    <FileText className="w-5 h-5 text-teal-400" />
                    <h3 className="text-sm font-bold tracking-tight">
                      Comprehensive Radiological Synthesis
                    </h3>
                  </div>

                  <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
                    {analysisResult.detailedAnalysis}
                  </p>

                  {/* Key Observations List */}
                  {analysisResult.keyObservations.length > 0 && (
                    <div className="pt-2 space-y-2">
                      <span className="text-xs font-bold text-teal-300 uppercase tracking-wider block">
                        Observed Diagnostic Highlights:
                      </span>
                      <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs text-slate-300">
                        {analysisResult.keyObservations.map((obs, idx) => (
                          <li key={idx} className="flex items-start space-x-2 bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/50">
                            <span className="text-teal-400 font-bold">•</span>
                            <span>{obs}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Recommendations */}
                  {analysisResult.recommendations && (
                    <div className="pt-3 border-t border-slate-800">
                      <h4 className="text-xs font-bold text-teal-300 mb-1 flex items-center space-x-1.5">
                        <ShieldCheck className="w-4 h-4 text-teal-400" />
                        <span>Recommended Clinical Pathway:</span>
                      </h4>
                      <p className="text-xs text-slate-300 leading-relaxed">
                        {analysisResult.recommendations}
                      </p>
                    </div>
                  )}
                </div>

                {/* 5. Action Bar */}
                <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2">
                  <button
                    onClick={handleReset}
                    className="w-full sm:w-auto bg-slate-900 hover:bg-slate-800 text-white text-xs font-bold px-6 py-3 rounded-xl transition-colors flex items-center justify-center space-x-2 cursor-pointer shadow-sm"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    <span>Analyze Another Image</span>
                  </button>

                  <button
                    onClick={() => window.print()}
                    className="w-full sm:w-auto border border-slate-200 text-slate-700 hover:bg-slate-50 text-xs font-semibold px-5 py-3 rounded-xl transition-colors flex items-center justify-center space-x-1.5 cursor-pointer"
                  >
                    <Printer className="w-3.5 h-3.5 text-slate-500" />
                    <span>Print Clinical Report</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
