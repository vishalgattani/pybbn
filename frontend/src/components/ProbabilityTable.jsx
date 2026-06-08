function pTrueColor(p) {
  if (p >= 0.7) return '#16a34a'
  if (p >= 0.4) return '#ca8a04'
  return '#dc2626'
}

export default function ProbabilityTable({ nodes }) {
  if (!nodes || nodes.length === 0) return null

  return (
    <div style={{ marginTop: '8px' }}>
      <h3 style={{ margin: '0 0 8px 0', fontSize: '16px' }}>Probability Table</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px' }}>
        <thead>
          <tr style={{ background: '#f1f3f5' }}>
            <th style={{ textAlign: 'left', padding: '8px', border: '1px solid #dee2e6' }}>Requirement</th>
            <th style={{ textAlign: 'center', padding: '8px', border: '1px solid #dee2e6' }}>P(True)</th>
            <th style={{ textAlign: 'center', padding: '8px', border: '1px solid #dee2e6' }}>P(False)</th>
          </tr>
        </thead>
        <tbody>
          {nodes.map((node, i) => {
            const pTrue = node.p_true ?? node.pTrue ?? 0
            const pFalse = node.p_false ?? node.pFalse ?? (1 - pTrue)
            const color = pTrueColor(pTrue)
            return (
              <tr key={i}>
                <td style={{ padding: '8px', border: '1px solid #dee2e6' }}>{node.name || node.requirement || `Node ${i + 1}`}</td>
                <td style={{ padding: '8px', textAlign: 'center', border: '1px solid #dee2e6', color, fontWeight: 600 }}>
                  {pTrue.toFixed(4)}
                </td>
                <td style={{ padding: '8px', textAlign: 'center', border: '1px solid #dee2e6', color: '#6c757d' }}>
                  {pFalse.toFixed(4)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
