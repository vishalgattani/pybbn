import { useState, useCallback } from 'react'
import SliderPanel from './components/SliderPanel'
import ProbabilityTable from './components/ProbabilityTable'
import AssuranceCaseSVG from './components/AssuranceCaseSVG'
import BeliefBarChart from './components/BeliefBarChart'

export default function App() {
  const [result, setResult] = useState(null)
  
  const handleCompute = useCallback(async (params) => {
    const res = await fetch('/api/bbn/compute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    })
    const data = await res.json()
    setResult(data)
  }, [])

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: '16px', padding: '16px', maxWidth: '1200px', margin: '0 auto' }}>
      <SliderPanel onCompute={handleCompute} />
      {result && (
        <>
          <AssuranceCaseSVG svg={result.assurance_case_svg} />
          <ProbabilityTable nodes={result.nodes} />
          <BeliefBarChart nodes={result.nodes} />
        </>
      )}
    </div>
  )
}
