import { useState, useEffect, useRef, useCallback } from 'react'
import * as RadixSlider from '@radix-ui/react-slider'

const SLIDER_DEFS = [
  { key: 'p_navigable', label: 'P(robot on navigable terrain)', min: 0, max: 1, step: 0.01, default: 0.9, int: false },
  { key: 'p_not_collide', label: 'P(robot not collide)', min: 0, max: 1, step: 0.01, default: 0.1, int: false },
  { key: 'p_pose_within', label: 'P(robot pose within region)', min: 0, max: 1, step: 0.01, default: 0.9, int: false },
  { key: 'nav_threshold', label: 'Nav Threshold', min: 0, max: 5, step: 1, default: 0, int: true },
  { key: 'collision_threshold', label: 'Collision Threshold', min: 0, max: 5, step: 1, default: 0, int: true },
  { key: 'pose_threshold', label: 'Pose Threshold', min: 0, max: 5, step: 1, default: 0, int: true },
]

function SliderRow({ def, value, onChange }) {
  const display = def.int ? value : value.toFixed(2)
  return (
    <div style={{ marginBottom: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', fontSize: '13px' }}>
        <span>{def.label}</span>
        <span style={{ fontFamily: 'monospace' }}>{display}</span>
      </div>
      <RadixSlider.Root
        style={{ position: 'relative', display: 'flex', alignItems: 'center', height: '20px' }}
        value={[value]}
        min={def.min}
        max={def.max}
        step={def.step}
        onValueChange={(v) => onChange(def.key, v[0])}
      >
        <RadixSlider.Track style={{ background: '#e2e2e2', position: 'relative', flexGrow: 1, height: '4px', borderRadius: '2px' }}>
          <RadixSlider.Range style={{ position: 'absolute', height: '100%', backgroundColor: '#2563eb', borderRadius: '2px' }} />
        </RadixSlider.Track>
        <RadixSlider.Thumb
          style={{
            display: 'block', width: '16px', height: '16px', backgroundColor: 'white',
            border: '2px solid #2563eb', borderRadius: '50%', cursor: 'pointer',
            boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
          }}
        />
      </RadixSlider.Root>
    </div>
  )
}

export default function SliderPanel({ onCompute }) {
  const [values, setValues] = useState(() => {
    const v = {}
    SLIDER_DEFS.forEach(d => v[d.key] = d.default)
    return v
  })

  const debounceRef = useRef(null)
  const onComputeRef = useRef(onCompute)
  useEffect(() => { onComputeRef.current = onCompute }, [onCompute])

  const handleChange = useCallback((key, val) => {
    setValues(prev => ({ ...prev, [key]: val }))

    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      setValues(current => {
        const params = { ...current }
        onComputeRef.current(params)
        return current
      })
    }, 300)
  }, [])

  return (
    <div style={{
      background: '#f8f9fa', border: '1px solid #dee2e6', borderRadius: '8px',
      padding: '16px', height: 'fit-content', position: 'sticky', top: '16px'
    }}>
      <h3 style={{ margin: '0 0 16px 0', fontSize: '16px', borderBottom: '1px solid #dee2e6', paddingBottom: '8px' }}>
        BBN Parameters
      </h3>
      {SLIDER_DEFS.map(def => (
        <SliderRow
          key={def.key}
          def={def}
          value={values[def.key]}
          onChange={handleChange}
        />
      ))}
      <p style={{ fontSize: '11px', color: '#6c757d', margin: '12px 0 0 0' }}>
        n_experiments: 5 (fixed)
      </p>
    </div>
  )
}
