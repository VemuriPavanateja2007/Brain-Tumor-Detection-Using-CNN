import React from 'react';

interface HealthHeartLogoProps {
  className?: string;
  size?: number;
  isSpinning?: boolean;
}

export const HealthHeartLogo: React.FC<HealthHeartLogoProps> = ({
  className = '',
  size = 180,
  isSpinning = false
}) => {
  return (
    <div
      className={`relative inline-flex items-center justify-center ${className}`}
      style={{ width: size, height: size }}
    >
      <svg
        viewBox="0 0 240 240"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="w-full h-full"
      >
        {/* Outer Segmented / Broken Circle matching the user's reference image */}
        <circle
          cx="120"
          cy="120"
          r="98"
          stroke="#0f172a"
          strokeWidth="11"
          strokeLinecap="round"
          strokeDasharray="280 28 80 28 14 28"
          className={isSpinning ? 'animate-[spin_10s_linear_infinite] origin-center' : ''}
        />

        {/* Outer decorative capsule dash matching the reference image */}
        <path
          d="M 205 78 A 98 98 0 0 1 213 98"
          stroke="#0f172a"
          strokeWidth="11"
          strokeLinecap="round"
        />

        {/* Outer decorative capsule dash at bottom left */}
        <path
          d="M 45 168 A 98 98 0 0 1 35 150"
          stroke="#0f172a"
          strokeWidth="11"
          strokeLinecap="round"
        />

        {/* Center Heart Contour with bold rounded stroke */}
        <path
          d="M120 188 C80 156 50 126 50 94 C50 68 70 50 94 50 C108 50 116 58 120 64 C124 58 132 50 146 50 C170 50 190 68 190 94 C190 126 160 156 120 188 Z"
          stroke="#0f172a"
          strokeWidth="11"
          strokeLinejoin="round"
          strokeLinecap="round"
          fill="none"
        />

        {/* Rounded Medical Plus / Cross Symbol in the center */}
        {/* Vertical Cross Bar */}
        <line
          x1="120"
          y1="96"
          x2="120"
          y2="148"
          stroke="#0f172a"
          strokeWidth="13"
          strokeLinecap="round"
        />

        {/* Horizontal Cross Bar */}
        <line
          x1="94"
          y1="122"
          x2="146"
          y2="122"
          stroke="#0f172a"
          strokeWidth="13"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
};
