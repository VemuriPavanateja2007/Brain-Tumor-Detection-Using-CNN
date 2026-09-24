import React from 'react';
import { Brain } from 'lucide-react';
import { MRIAnalyzerView } from './components/MRIAnalyzerView';

export default function App() {
  const h = React.createElement;
  return h('div', { className: 'min-h-screen bg-slate-50 flex flex-col font-sans text-slate-800 selection:bg-teal-100 selection:text-teal-900' },
    h('header', { className: 'sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-slate-200 shadow-2xs' },
      h('div', { className: 'max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between' },
        h('div', { className: 'flex items-center space-x-3' },
          h('div', { className: 'w-10 h-10 rounded-xl bg-slate-900 text-teal-400 flex items-center justify-center shadow-xs' },
            h(Brain, { className: 'w-5 h-5 text-teal-400' })),
          h('div', null,
            h('div', { className: 'flex items-center space-x-2' },
              h('span', { className: 'font-bold text-slate-900 text-base tracking-tight' }, 'Neuro MRI AI'),
              h('span', { className: 'text-[10px] font-semibold bg-teal-50 text-teal-800 border border-teal-200 px-2 py-0.5 rounded-md' }, 'AI Vision Analyzer')),
            h('p', { className: 'text-[11px] text-slate-500 font-normal hidden sm:block' }, 'Deep Brain MRI Scan Analysis'))))),
    h('main', { className: 'flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 py-6 sm:py-10' }, h(MRIAnalyzerView)),
    h('footer', { className: 'border-t border-slate-200 bg-white py-6 text-center text-xs text-slate-400 mt-12' },
      h('div', { className: 'max-w-5xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2' },
        h('p', null, `© ${new Date().getFullYear()} Neuro MRI AI — Intelligent Brain Scan Analysis.`),
        h('p', { className: 'text-slate-400 text-[11px]' }, 'Multimodal Image Vision Pipeline')))
  );
}
