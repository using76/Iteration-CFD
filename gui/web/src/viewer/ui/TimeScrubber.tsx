// Slider + play button over the dataset's time steps.
import { useEffect, useState } from 'react'
import type { TimeStepInfo } from '@cfd/shared'
import { IconPause, IconPlay } from './icons'

export function TimeScrubber({ times, index, onChange }: { times: TimeStepInfo[]; index: number; onChange: (i: number) => void }) {
  const [playing, setPlaying] = useState(false)
  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => onChange((index + 1) % times.length), 700)
    return () => clearInterval(id)
  }, [playing, index, times.length, onChange])
  const step = times[index]
  return (
    <div className="v3d-card pointer v3d-time" data-testid="viewer-time">
      <button className="v3d-iconbtn" onClick={() => setPlaying((p) => !p)} title={playing ? 'Pause' : 'Play'}>
        {playing ? <IconPause /> : <IconPlay />}
      </button>
      <input type="range" min={0} max={times.length - 1} step={1} value={index} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="label">
        {step ? `${step.label} (${index + 1}/${times.length})` : ''}
      </span>
    </div>
  )
}
