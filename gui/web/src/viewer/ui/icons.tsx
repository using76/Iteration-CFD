// Inline SVG icons for the viewer toolbar (no dependency on the shell's icon set).
import type { SVGProps } from 'react'

type P = SVGProps<SVGSVGElement> & { size?: number }

function Svg({ size = 16, children, ...rest }: P) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden {...rest}>
      {children}
    </svg>
  )
}

export const IconSelect = (p: P) => (
  <Svg {...p}>
    <path d="M4 3l7.5 17 2.4-6.6L20.5 11z" />
  </Svg>
)
export const IconPan = (p: P) => (
  <Svg {...p}>
    <path d="M12 3v18M3 12h18M12 3l-3 3M12 3l3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12l3 3M21 12l-3-3M21 12l-3 3" />
  </Svg>
)
export const IconRotate = (p: P) => (
  <Svg {...p}>
    <path d="M20 12a8 8 0 1 1-2.3-5.7" />
    <path d="M20 4v5h-5" />
  </Svg>
)
export const IconZoom = (p: P) => (
  <Svg {...p}>
    <circle cx="11" cy="11" r="7" />
    <path d="M21 21l-4.3-4.3M8 11h6M11 8v6" />
  </Svg>
)
export const IconSection = (p: P) => (
  <Svg {...p}>
    <path d="M4 20V8l8-4v12z" />
    <path d="M12 16l8-4V4" strokeDasharray="2 2" />
    <path d="M3 13l18-5" />
  </Svg>
)
export const IconSettings = (p: P) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z" />
  </Svg>
)
export const IconPlay = (p: P) => (
  <Svg {...p}>
    <path d="M7 4l12 8-12 8z" fill="currentColor" />
  </Svg>
)
export const IconPause = (p: P) => (
  <Svg {...p}>
    <path d="M7 4h4v16H7zM13 4h4v16h-4z" fill="currentColor" />
  </Svg>
)
export const IconEye = (p: P) => (
  <Svg {...p}>
    <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" />
    <circle cx="12" cy="12" r="3" />
  </Svg>
)
export const IconEyeOff = (p: P) => (
  <Svg {...p}>
    <path d="M3 3l18 18M10.6 10.6a3 3 0 0 0 4.2 4.2M6.6 6.6C4 8.3 2 12 2 12s3.5 7 10 7c1.9 0 3.6-.5 5-1.3M9.9 5.2A10 10 0 0 1 12 5c6.5 0 10 7 10 7a17 17 0 0 1-2.7 3.6" />
  </Svg>
)
export const IconTrash = (p: P) => (
  <Svg {...p}>
    <path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14M10 10v7M14 10v7" />
  </Svg>
)
export const IconCamera = (p: P) => (
  <Svg {...p}>
    <path d="M4 8h3l2-3h6l2 3h3v11H4z" />
    <circle cx="12" cy="13" r="3.5" />
  </Svg>
)
