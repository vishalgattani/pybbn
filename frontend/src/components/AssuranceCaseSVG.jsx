import { useState } from 'react'

export default function AssuranceCaseSVG({ svg }) {
  const [zoom, setZoom] = useState(1)

  if (!svg) return null

  return (
    <div style={{ marginTop: '8px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
        <h3 style={{ margin: 0, fontSize: '16px' }}>Assurance Case</h3>
        <button
          onClick={() => setZoom(z => Math.max(0.25, z - 0.1))}
          style={{ padding: '2px 8px', cursor: 'pointer', fontSize: '14px' }}
        >
          −
        </button>
        <span style={{ fontSize: '12px', fontFamily: 'monospace', minWidth: '40px', textAlign: 'center' }}>
          {Math.round(zoom * 100)}%
        </span>
        <button
          onClick={() => setZoom(z => Math.min(3, z + 0.1))}
          style={{ padding: '2px 8px', cursor: 'pointer', fontSize: '14px' }}
        >
          +
        </button>
        <button
          onClick={() => setZoom(1)}
          style={{ padding: '2px 8px', cursor: 'pointer', fontSize: '12px', color: '#6c757d' }}
        >
          Reset
        </button>
      </div>
      <div
        style={{
          border: '1px solid #dee2e6',
          borderRadius: '8px',
          overflow: 'auto',
          maxHeight: '600px',
          background: '#fff'
        }}
      >
        <div style={{ transform: `scale(${zoom})`, transformOrigin: 'top left', minWidth: 'fit-content' }}>
          <div dangerouslySetInnerHTML={{ __html: svg }} />
        </div>
      </div>
    </div>
  )
}
